import argparse
import threading

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TEMPLATE = """<Model Response>
{}
</Model Response>

<Reference Answer>
{}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output "True" . If it is incorrect or fails to select the correct option (if options are provided), output "False" . {}"""


class VerifyRequest(BaseModel):
    answer: str
    references: list[str]


def create_app(model_path: str) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/verify")
    def verify(request: VerifyRequest):
        prompts = [
            TEMPLATE.format(request.answer, reference, tokenizer.eos_token or "")
            for reference in request.references
        ]
        if not prompts:
            return {"prob": 0.0}
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        # ponytail: 串行推理优先保证单卡训练时 CPU 内存和延迟可预测。
        with lock, torch.inference_mode():
            logits = model(**inputs).logits.float()
        return {"prob": torch.softmax(logits, dim=-1)[:, 1].max().item()}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)
