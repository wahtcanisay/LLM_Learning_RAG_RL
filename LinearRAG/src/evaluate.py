import json
import os
from src.utils import normalize_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

class Evaluator:
    """评测最终生成答案，而不是直接评测检索排名。

    当前仓库提供两个 QA 指标：

    - LLM Accuracy：让另一个 LLM 判断预测与 gold answer 是否语义一致；
    - Contain Accuracy：标准化后，gold answer 是否是预测文本的子串。

    二者都不是 Recall@k、MRR 或 nDCG。后续医学迁移必须另外实现检索指标，
    不能把生成答案正确率当成证据召回率。
    """

    def __init__(self, llm_model, predictions_path):
        self.llm_model = llm_model
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        """读取 `qa()` 保存的逐题预测与 gold answer。"""
        prediction_results = json.load(open(self.predictions_path))
        return prediction_results
    
    def calculate_llm_accuracy(self,pre_answer,gold_ans):
        """调用 LLM Judge 做二分类语义正确性判断。"""
        system_prompt = """You are an expert evaluator. 
        """
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
        Generated answer: {pre_answer}
        Gold answer: {gold_ans}

        The generated answer should be considered correct if it:
        1. Contains the key information from the gold answer
        2. Is factually accurate and consistent with the gold answer
        3. Does not contain any contradicting information

        Respond with ONLY 'correct' or 'incorrect'.
        Response:
        """
        response = self.llm_model.infer([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        # 只接受精确的 "correct" 标签，其他格式一律记为错误。
        if response.strip().lower() == "correct":
            return 1.0
        else:
            return 0.0

    def calculate_contain(self,pre_answers,gold_ans):
        """计算标准化后的 gold-in-prediction 字符串包含指标。"""
        if pre_answers is None or pre_answers == "" or (isinstance(pre_answers, str) and pre_answers.strip() == ""):
            return 0            
        if gold_ans is None or gold_ans == "" or (isinstance(gold_ans, str) and gold_ans.strip() == ""):
            return 0
        s1 = normalize_answer(pre_answers)
        s2 = normalize_answer(gold_ans)
        if s2 in s1:
            return 1
        else:
            return 0
            
    def evaluate_sig_sample(self,idx,prediction):
        """评测一个样本，并保留原索引以便并发完成后写回正确位置。"""
        pre_answer = prediction["pred_answer"]
        gold_ans = prediction["gold_answer"]
        # llm_acc = 0.0
        llm_acc = self.calculate_llm_accuracy(pre_answer, gold_ans)
        contain_acc = self.calculate_contain(pre_answer, gold_ans)
        return idx, llm_acc, contain_acc

    def evaluate(self,max_workers):
        """并发评测全部预测，回写逐题字段并保存汇总结果。

        数据流：
            prediction
            → evaluate_sig_sample()
            → llm_accuracy + contain_accuracy
            → 回写 predictions.json
            → 聚合 evaluation_results.json
        """
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)
        
        # 并发的是 LLM Judge 请求；Contain Accuracy 本身只是本地字符串运算。
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx 
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            pbar = tqdm(total=len(futures), desc="Evaluating samples", unit="sample")
            for future in as_completed(futures):
                idx, llm_acc, contain_acc  = future.result()
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc
                # future 完成顺序可能与输入不同，idx 保证写回原问题。
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc
                total_llm_score += llm_acc
                total_contain_score += contain_acc
                completed += 1
                current_llm_acc = total_llm_score / completed
                current_contain_acc = total_contain_score / completed
                pbar.set_postfix({
                    'LLM_Acc': f'{current_llm_acc:.3f}',
                    'Contain_Acc': f'{current_contain_acc:.3f}'
                })
                pbar.update(1)
            pbar.close()

        llm_accuracy = sum(llm_scores) / len(llm_scores)
        contain_accuracy = sum(contain_scores) / len(contain_scores)

        logger.info(f"Evaluation Results:")
        logger.info(f"  LLM Accuracy: {llm_accuracy:.4f} ({sum(llm_scores)}/{len(llm_scores)})")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.4f} ({sum(contain_scores)}/{len(contain_scores)})")
        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)
        
        with open(os.path.join(os.path.dirname(self.predictions_path), "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump({"llm_accuracy": llm_accuracy, "contain_accuracy": contain_accuracy}, f, ensure_ascii=False, indent=4)
        return llm_accuracy, contain_accuracy
