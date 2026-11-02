"""CE 与离线 Top-32 蒸馏训练器"""

from transformers import Trainer

from .losses import forward_topk_loss


class DistillationTrainer(Trainer):
    """复用 Trainer 训练流程, 自定义 CE + 0.1 * Top-32 loss"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 使用每个 microbatch 的平均 loss, 不使用 num_items_in_batch
        self.model_accepts_loss_kwargs = False
        self._loss_metrics = {"ce_loss": 0.0, "top32_loss": 0.0}
        self._loss_microbatches = 0

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """单次学生前向, 计算 CE 与可选 Top-32 loss

        Args:
            model: 学生模型
            inputs: batch 字典
                input_ids, attention_mask, labels: [B, L] 张量
                target_positions: 可选, list[Tensor[N_i]]
                teacher_topk_ids: 可选, list[Tensor[N_i, 32]]
                teacher_topk_logprobs: 可选, list[Tensor[N_i, 32]]
            return_outputs: 是否同时返回模型输出
            num_items_in_batch: Trainer 传入的计数, 当前不使用
        Returns:
            False: 标量 loss
            True: (loss, outputs)
        """
        # 浅拷贝, 移除teacher字段内容, 兼容无教师缓存的CE数据
        model_inputs = inputs.copy()
        model_inputs.pop("teacher_topk_ids", None)
        model_inputs.pop("target_positions", None)
        model_inputs.pop("teacher_topk_logprobs", None)

        # student字段送入forward
        outputs = model(**model_inputs)
        ce_loss = outputs.loss

        # 存在教师缓存则加入蒸馏loss
        if "target_positions" in inputs:

            topk_loss = forward_topk_loss(
                outputs.logits,
                inputs["target_positions"],
                inputs["teacher_topk_ids"],
                inputs["teacher_topk_logprobs"],
            )
            loss = ce_loss + 0.1 * topk_loss

        else:
            loss = ce_loss

        # 分项日志只读取已计算的 loss，不增加前向或改变梯度。
        if model.training:
            self._loss_metrics["ce_loss"] += ce_loss.detach().item()
            if "target_positions" in inputs:
                self._loss_metrics["top32_loss"] += topk_loss.detach().item()
            self._loss_microbatches += 1

        if return_outputs:
            return (loss, outputs)
        else:
            return loss

    def log(self, logs, start_time=None):
        if "loss" in logs and self._loss_microbatches:
            logs = {**logs, **{
                name: value / self._loss_microbatches
                for name, value in self._loss_metrics.items()
            }}
            self._loss_metrics = {"ce_loss": 0.0, "top32_loss": 0.0}
            self._loss_microbatches = 0
        super().log(logs, start_time)
