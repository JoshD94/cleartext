# ClearText public demo

https://cleartext-ai2.pages.dev: paste up to 2,000 characters and watch the
default pipeline (`evals/default_pipeline.py`, `models/docs/DEFAULT_PIPELINE.md`)
simplify it step by step: F2 split, structure v4, learned words p50. No LLM.

```
browser ── Cloudflare Pages (site/public + site/functions/api/[[path]].js)
              │  KV "tunnel" = current *.trycloudflare.com URL
              ▼
        cloudflared quick tunnel (cleartext-tunnel.service, Yoga)
              ▼
        server.py on 127.0.0.1:8737 (cleartext-demo.service, Yoga)
              └─ child process: pipeline.DemoPipeline (loads on first job, exits after 15 idle min)
```

## Pieces

- `pipeline.py`: runs the three stages with the exact objects the default
  pipeline uses (`param_sweeps.SweepGenerator(...).predict(F2, text)`, then
  `DefaultAfterF2.structure.rewrite` and `.words.rewrite`) and builds the trace:
  per-step before/after, F2 edits by sub-stage, structure op labels, word swaps,
  FK grade before/after (`textstat`), and which numbers, negations and names
  survive. The study's disk memos are swapped for in-memory ones in this
  process only; no source under `models/src` or `evals/` is edited.
- `server.py`: stdlib HTTP, async jobs (`POST /api/simplify`, `GET /api/jobs/<id>`,
  `GET /api/health`), one serial worker, queue cap 6, two active jobs and 12
  submissions per 10 minutes per visitor IP, 2,000-character cap. Requests need
  the shared secret (`X-Demo-Key`), which only the Pages Function sends.
  Recent results (32, one hour) are kept in memory so example clicks return at once.
- `tunnel.py`: runs `cloudflared tunnel --url http://127.0.0.1:8737`, parses the
  quick-tunnel URL, waits until it answers, and registers it with
  `POST https://cleartext-ai2.pages.dev/api/_register` (same secret). It
  re-registers every 10 minutes; a cloudflared restart gives a new URL, which is
  registered again.
- `site/`: static page plus the Pages Function that proxies `/api/*` to the
  registered tunnel, and answers `{offline: true}` when nothing responds.

Privacy: submitted text and results stay in memory. Nothing is written to
disk (the child checks its scratch state stays empty), and logs carry only
method, path, timing and character counts.

## Run and restart (Yoga)

```sh
systemctl --user status cleartext-demo cleartext-tunnel
systemctl --user restart cleartext-demo      # unloads the models; next job is a cold start
systemctl --user restart cleartext-tunnel    # new tunnel URL, re-registered automatically
journalctl --user -u cleartext-tunnel -n 50  # shows "registered https://…trycloudflare.com"
```

Unit files live in `systemd/`; install with
`cp demo-web/systemd/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload &&
systemctl --user enable --now cleartext-demo cleartext-tunnel`. They run from
the primary checkout (`~/Documents/projects/cleartext`, with `models/.venv`,
`models/runs`, `models/data`), the parameter-sweep dev plan and the learned-word
state under `~/.local/state/cleartext/`. The secret is
`~/.config/cleartext-demo/secret` (mode 0600, never committed); the same value is
the Pages secret `DEMO_SECRET`. `cloudflared` is `~/.local/bin/cloudflared`
(official GitHub release, SHA-256 checked).

## Deploy the site (Mac; wrangler is logged in there only)

```sh
rsync -a --delete demo-web/site/ mac:scratch-cleartext-site/
ssh mac 'export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH CLOUDFLARE_ACCOUNT_ID=b2cb6710df5c0f4a84741bf194dd364d;
  cd ~/scratch-cleartext-site && npx -y wrangler@4 pages deploy --project-name cleartext --branch main --commit-dirty=true;
  rm -rf ~/scratch-cleartext-site'
# Rotate the secret: write a new ~/.config/cleartext-demo/secret, then
ssh mac '… npx -y wrangler@4 pages secret put DEMO_SECRET --project-name cleartext' < ~/.config/cleartext-demo/secret
```

KV namespace `cleartext-demo` (`61f84b30e14746d1bd9037d03f30610a`) is bound as
`DEMO_KV` in `site/wrangler.toml`.

## Checks

- `demo-web/test_server.py` (in `scripts/verify`): API, limits, trace shape,
  no text in logs, with the pipeline mocked.
- `tools/run-probe`: real pipeline on the examples through the job queue; checks
  the demo trace equals `F2 → DefaultAfterF2.__call__` and reports latency and peak RSS.
- `tools/shoot.mjs`: headless screenshots of the live site for the UI audit (`audit.md`).
