"""Demo server API and trace contract, with the pipeline mocked (no models load)."""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as DP  # noqa: E402
import server as SRV  # noqa: E402

SECRET = "test-secret"
TEXT = "The report was written by the committee. It was not finished in 2019, and Paris waited."


def fake_trace(text):
    f2 = {"output": text.replace(", and", ". And"), "stages": ["structure"],
          "edits": [{"stage": "structure", "rule": "learned t5", "before": text, "after": "x"}]}
    s_edits = [{"op": "passive", "rule": "by_agent", "before": "The report was written by the committee.",
                "after": "The committee wrote the report."}]
    after_s = f2["output"].replace("The report was written by the committee.", "The committee wrote the report.")
    return DP.build_trace(text, f2, after_s, s_edits, after_s, [], {"f2": 1.0, "structure": 0.5, "words": 0.2})


class FakeEngine:
    def __init__(self):
        self.loaded, self.alive, self.calls, self.stopped = False, False, [], 0
        self.gate = threading.Event()
        self.gate.set()

    def run(self, text, on_event, timeout=None):
        self.alive = True
        if not self.loaded:
            on_event("loading")
            self.loaded = True
        for s in DP.STAGES:
            on_event(s, {"key": "prev"} if s != "f2" else None)
        self.gate.wait(5)
        self.calls.append(len(text))
        if "boom" in text:
            raise RuntimeError("ValueError")
        return fake_trace(text)

    def stop(self):
        self.alive, self.loaded = False, False
        self.stopped += 1


@pytest.fixture
def api():
    engine = FakeEngine()
    jobs = SRV.Jobs(engine, idle_seconds=0.0)
    srv = SRV.make_server(0, jobs, SECRET)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None, key=SECRET, ip="1.2.3.4"):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"X-Demo-Key": key, "X-Demo-Client-IP": ip,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    yield call, jobs, engine
    srv.shutdown()


def test_requires_secret(api):
    call, _, _ = api
    assert call("/api/health", key="wrong")[0] == 403
    assert call("/api/simplify", {"text": TEXT}, key="")[0] == 403
    assert call("/api/health")[1]["ok"] is True


def test_job_lifecycle_and_trace(api):
    call, jobs, engine = api
    code, body = call("/api/simplify", {"text": TEXT})
    assert code == 202
    view = call(f"/api/jobs/{body['id']}")[1]
    assert view["status"] == "queued" and view["position"] == 1 and view["cold"] is True
    assert jobs.step() is True
    view = call(f"/api/jobs/{body['id']}")[1]
    assert view["status"] == "done"
    trace = view["trace"]
    assert [s["key"] for s in trace["steps"]] == ["f2", "structure", "words"]
    assert trace["input"] == TEXT and trace["output"] == trace["steps"][-1]["after"]
    f2, structure, words = trace["steps"]
    assert f2["changed"] and f2["groups"][0]["name"] == "Sentence split"
    assert structure["edits"][0]["label"] == "Passive to active" and structure["changed"]
    assert words["changed"] is False and words["edits"] == []
    assert set(trace["grade"]) == {"before", "after"}
    kept = trace["kept"]
    assert {"item": "2019", "kept": True} in kept["numbers"]
    assert all(i["item"] != "The" for i in kept["names"])
    assert {"item": "not", "kept": True} in kept["negations"]
    assert any(i["item"] == "Paris" for i in kept["names"])
    # The submitted text is dropped from the job once it runs.
    assert jobs.jobs[body["id"]]["text"] is None


def test_running_job_shows_finished_steps(api):
    call, jobs, engine = api
    jid = call("/api/simplify", {"text": TEXT})[1]["id"]
    engine.gate.clear()
    threading.Thread(target=jobs.step, daemon=True).start()
    for _ in range(100):
        view = call(f"/api/jobs/{jid}")[1]
        if view["status"] == "running" and len(view.get("steps", [])) == 2:
            break
        threading.Event().wait(0.02)
    assert view["stage"] == "words" and view["steps"] == [{"key": "prev"}, {"key": "prev"}]
    engine.gate.set()


def test_kept_names_skip_sentence_openers():
    kept = DP.kept_items("Applicants must apply. Marie Curie won. She met Paris officials.", "x")
    assert [n["item"] for n in kept["names"]] == ["Marie Curie", "Paris"]


def test_second_job_is_warm(api):
    call, jobs, _ = api
    call("/api/simplify", {"text": TEXT})
    jobs.step()
    jid = call("/api/simplify", {"text": TEXT})[1]["id"]
    jobs.step()
    assert call(f"/api/jobs/{jid}")[1]["cold"] is False


def test_limits(api, monkeypatch):
    call, jobs, _ = api
    assert call("/api/simplify", {"text": "x" * (SRV.MAX_CHARS + 1)})[0] == 413
    assert call("/api/simplify", {"text": "too short"})[1]["error"] == "too_short"
    assert call("/api/simplify", {"text": 5})[0] == 400
    # Two active jobs per IP, then a polite refusal; another IP still gets in.
    assert call("/api/simplify", {"text": TEXT})[0] == 202
    assert call("/api/simplify", {"text": TEXT})[0] == 202
    assert call("/api/simplify", {"text": TEXT}) == (429, {"error": "active"})
    assert call("/api/simplify", {"text": TEXT}, ip="5.6.7.8")[0] == 202
    monkeypatch.setattr(SRV, "QUEUE_MAX", 3)
    assert call("/api/simplify", {"text": TEXT}, ip="9.9.9.9") == (429, {"error": "busy"})
    monkeypatch.setattr(SRV, "QUEUE_MAX", 99)
    monkeypatch.setattr(SRV, "RATE_MAX", 3)
    while jobs.step():
        pass
    assert call("/api/simplify", {"text": TEXT})[0] == 202
    assert call("/api/simplify", {"text": TEXT}) == (429, {"error": "rate"})


def test_failure_hides_details_and_idle_unload(api):
    call, jobs, engine = api
    jid = call("/api/simplify", {"text": TEXT + " boom"})[1]["id"]
    jobs.step()
    view = call(f"/api/jobs/{jid}")[1]
    assert view["status"] == "failed" and view["error"] == "failed" and "trace" not in view
    assert jobs.step() is False and engine.stopped == 1  # idle_seconds=0: unloads at once


def test_no_text_in_logs(api, capsys):
    call, jobs, _ = api
    call("/api/simplify", {"text": TEXT})
    jobs.step()
    err = capsys.readouterr().err
    assert "committee" not in err and "Paris" not in err


def test_f2_grouping_and_labels():
    groups = DP.f2_edits({"stages": ["structure", "structure", "structure"],
                          "edits": [{"start": 0, "end": 11, "source": "in order to", "replacement": "to"},
                                    {"stage": "structure", "rule": "learned t5", "before": "a", "after": "b"},
                                    {"stage": "structure", "rule": "compound", "before": "c", "after": "d"}]})
    assert [(g["name"], g["count"]) for g in groups] == [("Sentence split", 2), ("Phrase rewrite", 1)]
    assert groups[1]["edits"] == [{"from": "in order to", "to": "to"}]
    labels = DP.structure_edits([{"op": op, "before": "a", "after": "b"}
                                 for op in ("passive", "appositive", "fronted", "nominal", "frames")])
    assert [e["label"] for e in labels] == ["Passive to active", "Side note made its own sentence",
                                            "Opening clause split off", "Noun turned into a verb", "Sentence rewrite"]


def test_repeat_text_is_served_from_memory(api):
    call, jobs, engine = api
    call("/api/simplify", {"text": TEXT})
    jobs.step()
    jid = call("/api/simplify", {"text": TEXT}, ip="7.7.7.7")[1]["id"]
    view = call(f"/api/jobs/{jid}")[1]
    assert view["status"] == "done" and view["trace"]["input"] == TEXT and len(engine.calls) == 1
