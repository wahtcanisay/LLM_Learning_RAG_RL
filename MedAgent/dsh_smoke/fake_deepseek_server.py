from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


PLANNER_JSON = '{"route":"SEARCH","search_query":"aspirin evidence"}'


class State:
    def __init__(self, mode: str, capture: Path, delay: float) -> None:
        self.mode = mode
        self.capture = capture
        self.delay = delay
        self.requests = 0
        self.lock = Lock()

    def record(self, request: object) -> int:
        with self.lock:
            self.requests += 1
            self.capture.parent.mkdir(parents=True, exist_ok=True)
            with self.capture.open("a", encoding="utf-8") as output:
                output.write(json.dumps(request, ensure_ascii=False) + "\n")
            return self.requests


class Handler(BaseHTTPRequestHandler):
    server: "Server"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        count = self.server.state.record(request)
        mode = self.server.state.mode
        if mode == "hang":
            time.sleep(self.server.state.delay)
            return
        if mode == "error" or (mode == "transient" and count == 1):
            self.send_response(401 if mode == "error" else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"synthetic smoke error"}}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        chunks = [
            {"id": "smoke", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": PLANNER_JSON}, "finish_reason": None}]},
            {"id": "smoke", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        if mode != "missing-usage":
            chunks[-1]["usage"] = {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26}
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: State) -> None:
        super().__init__(address, Handler)
        self.state = state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=("normal", "missing-usage", "error", "transient", "hang"), default="normal")
    parser.add_argument("--capture", default=".runtime/fake-model-requests.jsonl")
    parser.add_argument("--delay", type=float, default=30.0)
    args = parser.parse_args()
    server = Server(("127.0.0.1", args.port), State(args.mode, Path(args.capture), args.delay))
    print(f"fake DeepSeek service: http://127.0.0.1:{args.port}/v1 mode={args.mode}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
