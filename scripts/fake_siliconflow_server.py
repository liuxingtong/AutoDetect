"""HTTP(S) server that emulates the siliconflow /v1/chat/completions API.

Returns a deterministic JSON response based on prompt content.  This
is the minimum-fidelity mock needed to drive the full Adaptive run
end-to-end in a test, without touching the public internet.

Usage:
    python scripts/fake_siliconflow_server.py --port 18888 [--ssl]

With --ssl the server uses a self-signed certificate (generated in
/tmp/fake-sf-cert.pem) so the siliconflow client's https validation
passes.  In a real test, point the config at https://localhost:18888/v1
and disable cert verification in the underlying requests session.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _ensure_self_signed_cert(certfile: str, keyfile: str) -> None:
    if os.path.exists(certfile) and os.path.exists(keyfile):
        return
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-nodes", "-keyout", keyfile, "-out", certfile,
            "-days", "1", "-subj", "/CN=localhost",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _pick_response(prompt: str) -> str:
    """Map prompt content -> valid structured response.

    Same heuristic as tests/test_integration_chain.py::_pick_response
    so a real CLI run produces the same shape.
    """
    if "Examiner" in prompt or "selected_points" in prompt:
        return json.dumps(
            {
                "selected_points": [
                    {"group": "g", "subtask": "s", "knowledge_point": "p1"}
                ]
            }
        )
    if "questions" in prompt and "Return JSON only" in prompt:
        return json.dumps(
            {
                "questions": [
                    {"prompt": "Q1", "key_point": "p1"},
                ]
            }
        )
    if "knowledge point" in prompt and "Generate" in prompt:
        return json.dumps({"prompt": "Q2", "key_point": "p1"})
    if "Produce a correct" in prompt or "reference answer" in prompt.lower():
        return "Reference: 42"
    if "Rating: [[number]]" in prompt:
        return "Rating: [[3]]"
    if "is_weakness" in prompt or "Accept" in prompt or "discover" in prompt.lower():
        return json.dumps({"accept": False, "rationale": "no new mode"})
    if "violation" in prompt and "evidence" in prompt:
        return json.dumps(
            {
                "violation": "p1",
                "observed_behavior": "b",
                "trigger_context": "c",
                "evidence": "score=3",
            }
        )
    if "cluster" in prompt.lower() or "behavior" in prompt.lower():
        return json.dumps({"label": "different", "confidence": 0.7})
    if "promot" in prompt.lower():
        return json.dumps(
            {"decision": "promote", "confidence": 0.8, "reason": "ok"}
        )
    if "candidate" in prompt.lower() and "Generate" in prompt:
        return json.dumps(
            {
                "strategy": "verify",
                "objective": "obj",
                "reason": "recent failures",
                "candidate_budget": 1,
                "test_budget": 1,
            }
        )
    if "strategy" in prompt.lower() and "objective" in prompt.lower():
        # LLM controller prompt
        return json.dumps(
            {
                "strategy": "exploit",
                "objective": "obj",
                "reason": "r",
                "candidate_budget": 1,
                "test_budget": 1,
            }
        )
    return "{}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet the test server.
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        messages = payload.get("messages") or []
        prompt = messages[0]["content"] if messages else ""
        content = _pick_response(prompt)
        response = {
            "id": "fake-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }
        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--ssl", action="store_true",
        help="Wrap the socket in TLS using a self-signed cert."
    )
    parser.add_argument(
        "--cert", default="/tmp/fake-sf-cert.pem",
        help="Path for the self-signed cert (with --ssl)."
    )
    parser.add_argument(
        "--key", default="/tmp/fake-sf-key.pem",
        help="Path for the private key (with --ssl)."
    )
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.ssl:
        _ensure_self_signed_cert(args.cert, args.key)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    else:
        scheme = "http"
    print(
        "fake-siliconflow listening on {}://{}:{}".format(
            scheme, args.host, args.port
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
