"""Run a Cloudflare quick tunnel to the demo backend and register its URL.

Starts `cloudflared tunnel --url http://127.0.0.1:PORT`, reads the
*.trycloudflare.com URL from its log, waits until the backend answers through
the tunnel, then POSTs it to the Pages site's /api/_register with the shared
secret (stored in Cloudflare KV by the Pages Function). Re-registers every
REREGISTER seconds. Exits when cloudflared exits, so systemd restarts both and
the new URL is registered.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("DEMO_PORT", 8737))
SITE = os.environ.get("DEMO_SITE", "https://cleartext-ai2.pages.dev").rstrip("/")
SECRET = Path(os.environ.get("DEMO_SECRET_FILE", Path.home() / ".config/cleartext-demo/secret")).read_text().strip()
CLOUDFLARED = os.environ.get("CLOUDFLARED", str(Path.home() / ".local/bin/cloudflared"))
REREGISTER = 600
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, file=sys.stderr, flush=True)


def request(url, body=None, headers=None, timeout=15):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "cleartext-demo",
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def reachable(url):
    try:
        return request(url + "/api/health", headers={"X-Demo-Key": SECRET}) == 200
    except (urllib.error.URLError, OSError):
        return False


def register(url):
    return request(SITE + "/api/_register", {"url": url}, {"X-Demo-Key": SECRET}) == 200


def keep_registered(url, stop):
    for _ in range(90):  # DNS for a new quick tunnel takes a few seconds to minutes
        if stop.is_set() or reachable(url):
            break
        time.sleep(5)
    while not stop.is_set():
        try:
            register(url)
            log(f"registered {url} with {SITE}")
            stop.wait(REREGISTER)
        except (urllib.error.URLError, OSError) as exc:
            log(f"register failed: {exc}; retrying")
            stop.wait(15)


def main():
    proc = subprocess.Popen([CLOUDFLARED, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stop, url = threading.Event(), None
    for line in proc.stdout:
        sys.stderr.write(line)
        m = URL_RE.search(line)
        if m and url is None and "api.trycloudflare.com" not in m.group():
            url = m.group()
            threading.Thread(target=keep_registered, args=(url, stop), daemon=True).start()
    stop.set()
    sys.exit(proc.wait() or 1)


if __name__ == "__main__":
    main()
