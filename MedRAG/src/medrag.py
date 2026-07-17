import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem, DocExtracter
from template import *

from config import config

openai.api_type = openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv('OPENAI_API_KEY') or config["api_key"]

if openai.__version__.startswith("0"):
    openai.api_base = openai.api_base or os.getenv("OPENAI_API_BASE") or config.get("api_base")
    if openai.api_type == "azure":
        openai_client = lambda **x: openai.ChatCompletion.create(**{'engine' if k == 'model' else k: v for k, v in x.items()})["choices"][0]["message"]["content"]
    else:
        openai_client = lambda **x: openai.ChatCompletion.create(**x)["choices"][0]["message"]["content"]
else:
    if openai.api_type == "azure":
        openai.azure_endpoint = openai.azure_endpoint or os.getenv("OPENAI_ENDPOINT") or config.get("azure_endpoint")
        openai_client = lambda **x: openai.AzureOpenAI(
            api_version=openai.api_version,
            azure_endpoint=openai.azure_endpoint,
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content
    else:
        openai_client = lambda **x: openai.OpenAI(
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content

class MedRAG:
    # Day 1 代码入口：MedRAG 负责把“模型、检索器、Prompt、答案生成”串成一个调用接口。
    # 学习时先沿着 __init__ → RetrievalSystem → answer → generate 追踪数据流。

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", 
                 rag=True, 
                 follow_up=False, 
                 retriever_name="MedCPT", 
                 corpus_name="Textbooks", 
                 db_dir="./corpus", 
                 cache_dir=None, 
                 corpus_cache=False, 
                 HNSW=False):
        # 这些参数决定本次运行使用什么生成模型、是否检索、检索后端、语料库和索引目录。
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        # rag=True 会在初始化阶段创建 RetrievalSystem；它可能进一步下载语料或构建索引。
        # rag=False 不创建检索器，后面的 medrag_answer 会走 No-RAG（仅生成）路径。
        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, 
                                                    self.corpus_name, 
                                                    self.db_dir, 
                                                    cache=corpus_cache, 
                                                    HNSW=HNSW)
        else:
            self.retrieval_system = None
        # 模板只负责组织消息文本，不负责检索；真正渲染发生在 medrag_answer 中。
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        # 根据模型来源选择不同的 tokenizer、上下文长度和生成调用方式。
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "gemini" in self.llm_name.lower():
            import google.generativeai as genai
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel(
                model_name=self.llm_name.split('/')[-1],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                }
            )
            if "1.5" in self.llm_name.lower():
                self.max_length = 1048576
                self.context_length = 1040384
            else:
                self.max_length = 30720
                self.context_length = 28672
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            # 本地 Hugging Face 模型：先加载 tokenizer，再创建 Transformers pipeline。
            self.max_length = 2048
            self.context_length = 1024
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if "mixtral" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "llama-3" in llm_name.lower():
                self.max_length = 8192
                self.context_length = 7168
                if ".1" in llm_name or ".2" in llm_name:
                    self.max_length = 131072
                    self.context_length = 128000
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                # torch_dtype=torch.float16,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
            )
        
        self.follow_up = follow_up
        # follow_up=True 时把 answer 方法切换到 i-MedRAG 多轮追问流程；否则使用单轮流程。
        if self.rag and self.follow_up:
            self.answer = self.i_medrag_answer
            self.templates["medrag_system"] = simple_medrag_system
            self.templates["medrag_prompt"] = simple_medrag_prompt
            self.templates["i_medrag_system"] = i_medrag_system
            self.templates["follow_up_ask"] = follow_up_instruction_ask
            self.templates["follow_up_answer"] = follow_up_instruction_answer
        else:
            self.answer = self.medrag_answer

    def custom_stop(self, stop_str, input_len=0):
        # 为部分本地模型设置停止词，避免模型继续生成下一轮 User/Assistant 内容。
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages, **kwargs):
        '''
        generate response given messages
        '''
        # generate 是统一的生成出口：上层只准备 messages，不直接区分 API 或本地模型细节。
        if "openai" in self.llm_name.lower():
            ans = openai_client(
                model=self.model,
                messages=messages,
                temperature=0.0,
                **kwargs
            )
        elif "gemini" in self.llm_name.lower():
            response = self.model.generate_content(messages[0]["content"] + '\n\n' + messages[1]["content"], **kwargs)
            ans = response.candidates[0].content.parts[0].text
        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if "meditron" in self.llm_name.lower():
                # stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
            if "llama-3" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **kwargs
                )
            else:
                # self.model就是init当中定义好的pipeline
                response = self.model(
                    prompt,
                    do_sample=False, # 确定性采样，关闭随机采样
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **kwargs
                )
            # ans = response[0]["generated_text"]
            ans = response[0]["generated_text"][len(prompt):]
        return ans

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir = None, snippets=None, snippets_ids=None, **kwargs):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve

        rrf_k (int): Reciprocal Rank Fusion 的平滑常数，计算公式如下：
                        RRF_score += 1 / (rrf_k + rank + 1) rank越靠前贡献越大，
                        rrf_k越小，就强调第一名和前几名
                        rrf_k越大，不同排名之间的分数差距越小
                        
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''

        # 先把选项字典整理成 Prompt 中的多行文本，保证模型看到稳定的 A/B/C/... 顺序。
        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''

        # 第一步：取得证据。三种输入优先级是“直接给 snippets”→“给 snippets_ids”→“调用检索器”。
        if self.rag:
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                assert self.retrieval_system is not None
                retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)

            # 第二步：把结构化 snippet 转成带序号和标题的上下文字符串。
            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
            if len(contexts) == 0:
                contexts = [""]
            # 第三步：按不同 tokenizer 截断上下文，避免检索结果超过模型上下文窗口。
            if "openai" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            elif "gemini" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        else:
            # No-RAG 基线没有检索证据，contexts 为空，后面只渲染 CoT Prompt。
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 第四步：将 context/question/options 渲染成 messages，再交给统一的 generate()。
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            ans = self.generate(messages, **kwargs)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options)
                messages=[
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages, **kwargs)
                answers.append(re.sub("\s+", " ", ans))
        
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers[0] if len(answers)==1 else answers, retrieved_snippets, scores

    def i_medrag_answer(self, question, options=None, k=32, rrf_k=100, save_path = None, n_rounds=4, n_queries=3, qa_cache_path=None, **kwargs):
        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''
        QUESTION_PROMPT = f"Here is the question:\n{question}\n\n{options}"

        context = ""
        qa_cache = []
        if qa_cache_path is not None and os.path.exists(qa_cache_path):
            qa_cache = eval(open(qa_cache_path, 'r').read())[:n_rounds]
            if len(qa_cache) > 0:
                context = qa_cache[-1]
            n_rounds = n_rounds - len(qa_cache)
        last_context = None

        # Run in loop
        max_iterations = n_rounds + 3
        saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]

        for i in range(max_iterations):
            if i < n_rounds:
                if context == "":
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
                else:                
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
            elif context != last_context:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            elif len(messages) == 1:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            saved_messages.append(messages[-1])
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
            last_context = context
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)       
            if i >= n_rounds and ("## Answer" in last_content or "answer is" in last_content.lower()):
                messages.append(response_message)
                messages.append(
                    {
                        "role": "user",
                        "content": "Output the answer in JSON: {'answer': your_answer (A/B/C/D)}" if options else "Output the answer in JSON: {'answer': your_answer}",
                    }
                )
                saved_messages.append(messages[-1])
                answer_content = self.generate(messages, **kwargs)
                answer_message = {"role": "assistant", "content": answer_content}
                messages.append(answer_message)
                saved_messages.append(messages[-1])
                if save_path:
                    with open(save_path, 'w') as f:
                        json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
                return messages[-1]["content"], messages
            elif "## Queries" in last_content:
                messages = messages[:-1]
                if last_content.split("## Queries")[-1].strip() == "":
                    print("Empty queries. Continue with next iteration.")
                    continue
                try:
                    action_str = self.generate([
                        {
                            "role": "user",
                            "content": f"Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{\"output\": [\"query 1\", ..., \"query N\"]}}",
                        }
                    ], **kwargs)
                    action_str = re.search(r"output\": (\[.*\])", action_str, re.DOTALL).group(1)
                    action_list = [re.sub(r'^\d+\.\s*', '', s.strip()) for s in eval(action_str)]
                except Exception as E:
                    print("Error parsing action list. Continue with next iteration.")
                    error_class = E.__class__.__name__
                    error = f"{error_class}: {str(E)}"
                    print(error)
                    if save_path:
                        with open(save_path + ".error", 'a') as f:
                            f.write(f"{error}\n")                
                    continue
                for question in action_list:
                    if question.strip() == "":
                        continue
                    try:
                        rag_result = self.medrag_answer(question, k=k, rrf_k=rrf_k, **kwargs)[0]
                        context += f"\n\nQuery: {question}\nAnswer: {rag_result}"
                        context = context.strip()
                    except Exception as E:
                        error_class = E.__class__.__name__
                        error = f"{error_class}: {str(E)}"
                        print(error)
                        if save_path:
                            with open(save_path + ".error", 'a') as f:
                                f.write(f"{error}\n")
                qa_cache.append(context)
                if qa_cache_path:
                    with open(qa_cache_path, 'w') as f:
                        json.dump(qa_cache, f, indent=4)
            else:
                messages.append(response_message)
                print("No queries or answer. Continue with next iteration.")
                continue
        return messages[-1]["content"], messages

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)
