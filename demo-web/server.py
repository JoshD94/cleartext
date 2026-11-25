"""ClearText public demo backend: a stdlib HTTP server with an async job API.

  POST /api/simplify {"text": ...}  -> {"id": ...}
  GET  /api/jobs/<id>               -> {"status", "position", "stage", "cold", "trace"?}
  GET  /api/health                  -> {"ok", "warm", "queue"}

Every request must carry the shared secret in X-Demo-Key (the Cloudflare Pages
Function adds it, with the visitor's IP in X-Demo-Client-IP). Jobs run one at a
time in a child process that loads the models on first use and exits after
IDLE_SECONDS without work, so the RAM goes back to the host. Submitted text and
results live only in memory: nothing is written to disk and logs carry no text.
"""

import hmac
import json
import multiprocessing as mp
import os
import secrets
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_CHARS = int(os.environ.get("DEMO_MAX_CHARS", 2000))
MIN_WORDS = 3
QUEUE_MAX = int(os.environ.get("DEMO_QUEUE_MAX", 6))
PER_IP_ACTIVE = 2
RATE_WINDOW, RATE_MAX = 600.0, int(os.environ.get("DEMO_RATE_MAX", 12))
IDLE_SECONDS = float(os.environ.get("DEMO_IDLE_SECONDS", 900))
JOB_TTL = 900.0
CACHE_MAX, CACHE_TTL = 32, 3600.0
JOB_TIMEOUT = float(os.environ.get("DEMO_JOB_TIMEOUT", 900))
THREADS = int(os.environ.get("DEMO_THREADS", 4))
SECRET_FILE = Path(os.environ.get("DEMO_SECRET_FILE", Path.home() / ".config/cleartext-demo/secret"))


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, file=sys.stderr, flush=True)


def child_main(conn, threads):
    """Loads the pipeline lazily and serves one text at a time over a pipe."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    pipe = None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg.get("cmd") == "exit":
            return
        try:
            if pipe is None:
                conn.send({"event": "stage", "stage": "loading"})
                t = time.monotonic()
                import pipeline
                pipe = pipeline.DemoPipeline(threads=threads)
                conn.send({"event": "loaded", "seconds": round(time.monotonic() - t, 1)})
            trace = pipe.run(msg["text"], lambda s, done=None: conn.send({"event": "stage", "stage": s, "done": done}))
            conn.send({"event": "done", "trace": trace})
        except Exception as exc:
            conn.send({"event": "error", "kind": type(exc).__name__})


class Engine:
    """Real pipeline runner: a child process, restarted on demand."""

    def __init__(self, threads=THREADS):
        self.threads, self.proc, self.conn, self.loaded = threads, None, None, False

    @property
    def alive(self):
        return self.proc is not None and self.proc.is_alive()

    def start(self):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=child_main, args=(child, self.threads), daemon=True)
        self.proc.start()
        child.close()
        self.loaded = False
        log(f"engine started pid={self.proc.pid}")

    def stop(self):
        if self.proc is None:
            return
        try:
            self.conn.send({"cmd": "exit"})
        except (OSError, BrokenPipeError):
            pass
        self.proc.join(20)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(5)
        log(f"engine stopped pid={self.proc.pid}")
        self.proc, self.conn, self.loaded = None, None, False

    def run(self, text, on_event, timeout=JOB_TIMEOUT):
        if not self.alive:
            self.start()
        self.conn.send({"text": text})
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self.stop()
                raise TimeoutError
            if not self.conn.poll(min(left, 5)):
                if not self.alive:
                    self.proc, self.loaded = None, False
                    raise RuntimeError("engine exited")
                continue
            try:
                msg = self.conn.recv()
            except EOFError:
                self.stop()
                raise RuntimeError("engine exited")
            if msg["event"] == "loaded":
                self.loaded = True
                log(f"engine loaded in {msg['seconds']}s")
            elif msg["event"] == "stage":
                on_event(msg["stage"], msg.get("done"))
            elif msg["event"] == "done":
                return msg["trace"]
            elif msg["event"] == "error":
                raise RuntimeError(msg["kind"])


class Jobs:
    def __init__(self, engine, idle_seconds=IDLE_SECONDS):
        self.engine, self.idle_seconds = engine, idle_seconds
        self.lock = threading.Lock()
        self.jobs, self.pending = {}, deque()
        self.wake = threading.Event()
        self.hits = {}
        self.last_work = time.monotonic()
        self.stats = {"done": 0, "failed": 0, "cold_seconds": [], "warm_seconds": []}
        self.cache = {}

    def _limited(self, ip, now):
        hits = [t for t in self.hits.get(ip, []) if now - t < RATE_WINDOW]
        self.hits[ip] = hits
        if len(hits) >= RATE_MAX:
            return "rate"
        active = sum(1 for j in self.jobs.values() if j["ip"] == ip and j["status"] in ("queued", "running"))
        if active >= PER_IP_ACTIVE:
            return "active"
        return None

    def submit(self, text, ip):
        now = time.monotonic()
        with self.lock:
            self._expire(now)
            if len(self.pending) >= QUEUE_MAX:
                return None, "busy"
            why = self._limited(ip, now)
            if why:
                return None, why
            self.hits[ip].append(now)
            jid = secrets.token_urlsafe(12)
            cached = self.cache.get(text)
            if cached and now - cached[0] < CACHE_TTL:
                self.jobs[jid] = {"id": jid, "ip": ip, "status": "done", "stage": "done", "text": None,
                                  "created": now, "started": now, "finished": now, "trace": cached[1],
                                  "error": None, "cold": False}
                return jid, None
            self.jobs[jid] = {"id": jid, "ip": ip, "status": "queued", "stage": None, "text": text,
                              "created": now, "started": None, "finished": None, "trace": None,
                              "error": None, "cold": not self.engine.loaded}
            self.pending.append(jid)
        self.wake.set()
        return jid, None

    def view(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if job is None:
                return None
            now = time.monotonic()
            out = {"id": jid, "status": job["status"], "stage": job["stage"], "cold": job["cold"],
                   "position": (list(self.pending).index(jid) + 1) if jid in self.pending else 0,
                   "elapsed": round((job["finished"] or now) - (job["started"] or now), 1)}
            if job["status"] == "running":
                out["steps"] = list(job.get("steps", []))
            if job["status"] == "done":
                out["trace"] = job["trace"]
            if job["status"] == "failed":
                out["error"] = job["error"]
            return out

    def health(self):
        with self.lock:
            return {"ok": True, "warm": self.engine.loaded, "queue": len(self.pending),
                    "running": any(j["status"] == "running" for j in self.jobs.values())}

    def _expire(self, now):
        for jid in [j for j, v in self.jobs.items() if v["finished"] and now - v["finished"] > JOB_TTL]:
            del self.jobs[jid]
        for ip in [ip for ip, h in self.hits.items() if not h or now - h[-1] > RATE_WINDOW]:
            del self.hits[ip]
        for text in [t for t, (at, _) in self.cache.items() if now - at > CACHE_TTL]:
            del self.cache[text]

    def step(self):
        """Run the next job, or unload when idle. Returns True if a job ran."""
        with self.lock:
            jid = self.pending.popleft() if self.pending else None
            if jid is not None:
                job = self.jobs[jid]
                job["status"], job["started"] = "running", time.monotonic()
                job["cold"] = not self.engine.loaded
                job["stage"] = "loading" if job["cold"] else "f2"
                job["steps"] = []
                text, job["text"] = job["text"], None
        if jid is None:
            if self.engine.alive and time.monotonic() - self.last_work > self.idle_seconds:
                self.engine.stop()
            return False

        def on_stage(stage, done=None):
            with self.lock:
                job["stage"] = stage
                if done is not None:
                    job["steps"].append(done)

        t = time.monotonic()
        try:
            trace = self.engine.run(text, on_stage)
            del text
            with self.lock:
                job.update(status="done", trace=trace, stage="done", finished=time.monotonic())
                self.cache[trace["input"]] = (time.monotonic(), trace)
                while len(self.cache) > CACHE_MAX:
                    del self.cache[next(iter(self.cache))]
                self.stats["done"] += 1
                self.stats["cold_seconds" if job["cold"] else "warm_seconds"].append(round(time.monotonic() - t, 1))
            log(f"job done cold={job['cold']} seconds={time.monotonic() - t:.1f} chars={len(trace['input'])}")
        except Exception as exc:
            with self.lock:
                job.update(status="failed", finished=time.monotonic(),
                           error="timeout" if isinstance(exc, TimeoutError) else "failed")
                self.stats["failed"] += 1
            log(f"job failed kind={type(exc).__name__}: {str(exc)[:60] if isinstance(exc, RuntimeError) else ''}")
        self.last_work = time.monotonic()
        return True

    def loop(self):
        while True:
            if not self.step():
                self.wake.wait(30)
                self.wake.clear()


def read_secret():
    return SECRET_FILE.read_text().strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "cleartext-demo"
    jobs = None
    secret = ""

    def log_message(self, fmt, *args):
        path = self.path.split("?")[0]
        if path.startswith("/api/jobs/"):
            return
        log(f"{self.command} {path} {args[1] if len(args) > 1 else ''}")

    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        key = self.headers.get("X-Demo-Key", "")
        return bool(self.secret) and hmac.compare_digest(key.encode(), self.secret.encode())

    def _ip(self):
        return (self.headers.get("X-Demo-Client-IP") or self.client_address[0])[:64]

    def do_GET(self):
        if not self._authorized():
            return self._json(403, {"error": "forbidden"})
        path = self.path.split("?")[0]
        if path == "/api/health":
            return self._json(200, self.jobs.health())
        if path.startswith("/api/jobs/"):
            view = self.jobs.view(path[len("/api/jobs/"):])
            return self._json(200, view) if view else self._json(404, {"error": "unknown"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            return self._json(403, {"error": "forbidden"})
        if self.path.split("?")[0] != "/api/simplify":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_CHARS * 6 + 200:
            return self._json(413, {"error": "too_long", "max": MAX_CHARS})
        try:
            text = json.loads(self.rfile.read(length))["text"]
        except (ValueError, KeyError, TypeError):
            return self._json(400, {"error": "bad_request"})
        if not isinstance(text, str):
            return self._json(400, {"error": "bad_request"})
        text = text.replace("\r\n", "\n").strip()
        if len(text) > MAX_CHARS:
            return self._json(413, {"error": "too_long", "max": MAX_CHARS})
        if len(text.split()) < MIN_WORDS:
            return self._json(400, {"error": "too_short"})
        jid, why = self.jobs.submit(text, self._ip())
        if jid is None:
            return self._json(429, {"error": why})
        return self._json(202, {"id": jid})


def make_server(port, jobs, secret):
    handler = type("H", (Handler,), {"jobs": jobs, "secret": secret})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main():
    port = int(os.environ.get("DEMO_PORT", 8737))
    jobs = Jobs(Engine())
    threading.Thread(target=jobs.loop, daemon=True).start()
    server = make_server(port, jobs, read_secret())
    log(f"listening on 127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
