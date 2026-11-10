"""Tests for the Week 7 ablation ladder's HTTP transport.

The mock server mimics models/demo/server.py's /api/simplify contract:
it records the request fields and marks the output with the stages the
request enabled, so the tests verify the adapter sends the right toggles
(structure/phrases booleans, max_edits=0 to silence word edits) and that
full_no_guardrails fails loudly over HTTP instead of running guarded.

Run:  python -m unittest evals.test_adapter_ladder -v   (from repo root)
Stdlib only.
"""

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.model_adapter import (  # noqa: E402
    ABLATION_SYSTEMS,
    ModelUnavailable,
    make_ablation_systems,
    register_ablation_systems,
)
from evals.runner import SYSTEMS  # noqa: E402


class MockDemoServer:
    """Records requests; echoes which stages the request enabled."""

    def __init__(self):
        self.requests = []

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(
                    int(self.headers["Content-Length"])).decode("utf-8"))
                outer.requests.append(payload)
                marks = []
                if payload.get("structure"):
                    marks.append("[S]")
                if payload.get("phrases"):
                    marks.append("[P]")
                if payload.get("max_edits", 0) > 0:
                    marks.append("[W]")
                body = json.dumps(
                    {"output": payload["text"] + " " + " ".join(marks)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class LadderHttpTest(unittest.TestCase):
    def test_each_arm_sends_its_toggles(self):
        with MockDemoServer() as url:
            systems = make_ablation_systems(url=url)
            self.assertEqual(set(systems), set(ABLATION_SYSTEMS))
            expected = {
                "lexical_only": "[W]",
                "phrase_only": "[P]",
                "structure_only": "[S]",
                "lexical_structural": "[S] [W]",
                "full_guarded": "[S] [P] [W]",
            }
            for name, marks in expected.items():
                out = systems[name]("hello world")
                self.assertTrue(out.endswith(marks),
                                f"{name}: got {out!r}, want suffix {marks!r}")

    def test_phrase_and_structure_arms_silence_word_edits(self):
        server = MockDemoServer()
        with server as url:
            systems = make_ablation_systems(url=url)
            systems["phrase_only"]("x")
            systems["structure_only"]("x")
            for req in server.requests:
                self.assertEqual(req["max_edits"], 0, req)
            self.assertFalse(server.requests[0]["structure"])
            self.assertTrue(server.requests[0]["phrases"])
            self.assertTrue(server.requests[1]["structure"])
            self.assertFalse(server.requests[1]["phrases"])

    def test_no_guardrails_fails_loudly_over_http(self):
        with MockDemoServer() as url:
            systems = make_ablation_systems(url=url)
            with self.assertRaises(ModelUnavailable):
                systems["full_no_guardrails"]("hello world")

    def test_register_ladder_registers_all_names(self):
        with MockDemoServer() as url:
            register_ablation_systems(url=url)
            for name in ABLATION_SYSTEMS:
                self.assertIn(name, SYSTEMS)


if __name__ == "__main__":
    unittest.main()
