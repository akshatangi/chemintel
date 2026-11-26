"""Starts a local mock OpenAI-compatible server and checks the client sends the retrieved passages and parses the reply.
Proves request/response handling only. It says nothing about any real model."""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chemintel.generators import OpenAICompatGenerator, build_rag_prompt

seen = {}


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen["path"], seen["body"] = self.path, body
        out = json.dumps({"choices": [{"message": {"content": "MOCK ANSWER [1]"}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a): pass


def test_client():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    g = OpenAICompatGenerator("qwen2.5:1.5b", f"http://127.0.0.1:{srv.server_port}/v1")
    prompt = build_rag_prompt("Why does salt raise boiling point?", [{"text": "Boiling point elevation passage."}])
    assert g.generate(prompt) == "MOCK ANSWER [1]"
    assert seen["path"] == "/v1/chat/completions"
    assert "Boiling point elevation passage." in seen["body"]["messages"][0]["content"]
    assert seen["body"]["model"] == "qwen2.5:1.5b"
    srv.shutdown()


if __name__ == "__main__":
    test_client(); print("ok test_client")
