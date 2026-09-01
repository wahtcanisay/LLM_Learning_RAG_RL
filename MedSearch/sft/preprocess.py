"""MedSearch SFT sample encoders.

The surrounding training and tests are maintained by the teacher. The two
encoding functions are the student-owned implementation boundary.
"""

IGNORE_INDEX = -100
OPTION_LABELS = ("A", "B", "C", "D")


def format_stage_b_prompt(sample: dict) -> str:
    """Build the single user prompt shared by Stage-B SFT and later RL."""
    question = sample["question"]
    options = sample["options"]
    if not isinstance(question, str) or not question:
        raise ValueError("Stage-B question must be a non-empty string")
    if tuple(options) != OPTION_LABELS:
        raise ValueError("Stage-B options must contain A, B, C and D in order")
    if any(not isinstance(options[label], str) or not options[label] for label in OPTION_LABELS):
        raise ValueError("Stage-B option values must be non-empty strings")

    instruction = (
        "Answer the given multiple-choice question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between "
        "<information> and </information>. "
        "You can search as many times as you want. "
        "If you find no further external knowledge needed, provide only the option letter inside "
        "<answer> and </answer>, for example <answer>B</answer>."
    )
    choices = "\n".join(f"{label}. {options[label]}" for label in OPTION_LABELS)
    return f"{instruction}\nQuestion: {question}\nOptions:\n{choices}\n"


def encode_stage_a(sample: dict, tokenizer, max_length: int) -> dict[str, list[int]]:
    """Encode one normal Stage-A chat sample.

    Contract:
    - Read ``sample["messages"]`` in its original order.
    - Reuse ``tokenizer.apply_chat_template`` for Qwen2.5 formatting.
    - Keep system/user tokens in ``input_ids`` but set their labels to -100.
    - Train on assistant content and each assistant turn's EOS token.
    - Return equally sized ``input_ids``, ``attention_mask`` and ``labels``.
    - Do not silently return a sequence longer than ``max_length``.

    Student task: implement this function without adding a Dataset or Trainer.
    """
    # Student task: replace this placeholder with the Stage-A encoding logic.
    messages = sample["messages"]
    eos_id = tokenizer.eos_token_id # eosid，整数

    # 结果列表
    input_ids = []
    attention_mask = []
    labels = []

    accumulated = [] # 当前对话
    previous_rendered = "" # 前一轮对话内容chat_template版

    for message in messages:
        if message["role"] != "assistant":
            accumulated.append(message) # 当前为system/user直接加入
            continue

        rendered = tokenizer.apply_chat_template(accumulated,
                                                 tokenize = False,
                                                 add_generation_prompt=True
                                                 )

        source_text = rendered[len(previous_rendered):] # 切割上一轮对话
        target_text = message["content"]

        source_ids = tokenizer.encode(source_text,
                                      add_special_tokens = False)
        target_ids = tokenizer.encode(target_text,
                                      add_special_tokens = False)

        input_ids.extend(source_ids)
        labels.extend([IGNORE_INDEX] * len(source_ids))

        input_ids.extend(target_ids)
        labels.extend(target_ids)

        input_ids.append(eos_id)
        labels.append(eos_id)


        accumulated.append(message) # assistant对话加入
        previous_rendered = tokenizer.apply_chat_template(accumulated,
                                                 tokenize = False,
                                                 add_generation_prompt=False
                                                 ) # 本轮对话加入更新

        # 检查是否过长
        if len(input_ids) > max_length:
            raise ValueError(
                f"encoded sample has {len(input_ids)} tokens, "
                f"exceeding max_length={max_length}"
            )

        # A中对话均加入attention计算
        attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def encode_stage_b(sample: dict, tokenizer, max_length: int) -> dict[str, list[int]]:
    """Encode one Search100 trajectory in the exact Search-R1 rolling format.

    Contract:
    - Render the Qwen system/user prompt once at the beginning.
    - Train on think/search/answer tokens.
    - Keep information tokens in context with labels set to -100.
    - Do not add an EOS after a search action; end only after the final answer.
    - Return equally sized ``input_ids``, ``attention_mask`` and ``labels``.

    Serialization:
    - Initial input: one user message containing ``format_stage_b_prompt(sample)``.
    - Search: ``<think>{thinking}</think>\n<search>{query}</search>``.
    - Information: ``\n\n<information>{content}</information>\n\n``.
    - Answer: ``<think>{thinking}</think>\n<answer>{answer}</answer>`` plus EOS.

    Student task: implement this function without changing the frozen JSON or
    adding new tokenizer tokens.
    """
    prompt = format_stage_b_prompt(sample) # GPT编写的工程函数， 将协议说明+题干+选项进行合并
    trajectory = sample["trajectory"]
    eos_id = tokenizer.eos_token_id

    input_ids = []
    labels = []

    initial_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize = False,
                                                 add_generation_prompt=True)
    initial_text_ids = tokenizer.encode(initial_text,
                                        add_special_tokens = False)

    input_ids.extend(initial_text_ids)
    labels.extend([IGNORE_INDEX] * len(initial_text_ids))

    for step in trajectory:
        if step["type"] == "search":
            # search：模型自己生成，因此训练
            text = (
                f"<think>{step['thinking']}</think>\n"
                f"<search>{step['query']}</search>"
            )

        elif step["type"] == "information":
            # information：环境插入，因此只作为上下文，不训练
            text = f"\n\n<information>{step['content']}</information>\n\n"

        elif step["type"] == "answer":
            # answer：模型自己生成，因此训练；随后才结束
            text = (
                f"<think>{step['thinking']}</think>\n"
                f"<answer>{step['answer']}</answer>"
            )

        else:
            raise ValueError(
                f"unknown trajectory step type: {step["type"]}"
            )

        text_ids = tokenizer.encode(text,
                                    add_special_tokens = False)
        input_ids.extend(text_ids)

        if step["type"] != "information":
            labels.extend(text_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(text_ids))

        if step["type"] == "answer":
            input_ids.append(eos_id)
            labels.append(eos_id)
            break

    # B中对话均加入attention计算
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
