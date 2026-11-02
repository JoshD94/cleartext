"""Resumable local evaluation cells, bound to a frozen experiment signature.

Raw text belongs in an external state directory. The caller serializes writers
and verifies that the signature covers its sources, inputs, models and settings.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ReplayStore:
    def __init__(self, directory, signature):
        self.root = Path(directory)
        self.signature = digest(signature)
        manifest = self.root / "manifest.json"
        expected = {"signature": self.signature, "experiment": signature}
        if manifest.exists():
            if json.loads(manifest.read_text()) != expected:
                raise ValueError("Checkpoint experiment signature changed")
        else:
            write_json(manifest, expected)

    def _identity(self, passage, system):
        return {"signature": self.signature, "id": passage["id"],
                "system": system, "source_sha256": digest(passage["original"])}

    def _path(self, passage, system, stage):
        return self.root / stage / (digest([passage["id"], system]) + ".json")

    def _read(self, passage, system, stage):
        path = self._path(passage, system, stage)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value.get("identity") != self._identity(passage, system):
            raise ValueError(f"Checkpoint identity changed: {passage['id']} / {system}")
        if not isinstance(value.get("output"), str):
            raise ValueError("Invalid checkpoint output")
        if stage == "cells" and not all(isinstance(value.get(k), dict)
                                        for k in ("readability", "preservation")):
            raise ValueError("Incomplete checkpoint metrics")
        return value

    def completed(self, passage, system):
        return self._read(passage, system, "cells") is not None

    def evaluate(self, passage, system, predict, readability, preservation, progress=print):
        entry = self._read(passage, system, "cells")
        if entry is not None:
            progress(f"cached {passage['id']} {system}")
            return entry
        entry = self._read(passage, system, "outputs")
        if entry is None:
            progress(f"predict {passage['id']} {system}")
            output = predict(passage["original"])
            if not isinstance(output, str):
                raise ValueError("System returned a non-string output")
            entry = {"identity": self._identity(passage, system), "output": output}
            write_json(self._path(passage, system, "outputs"), entry)
        key = digest([passage["original"], entry["output"]])
        path = self.root / "metrics" / (key + ".json")
        pair = {"source": passage["original"], "output": entry["output"]}
        if path.exists():
            metrics = json.loads(path.read_text())
            if metrics.get("pair") != pair or metrics.get("signature") != self.signature:
                raise ValueError("Metric cache identity changed")
        else:
            progress(f"score {passage['id']} {system}")
            metrics = {"signature": self.signature, "pair": pair,
                       "readability": readability(entry["output"]),
                       "preservation": preservation(passage["original"], entry["output"])}
            write_json(path, metrics)
        entry.update(readability=metrics["readability"], preservation=metrics["preservation"])
        write_json(self._path(passage, system, "cells"), entry)
        progress(f"saved {passage['id']} {system}")
        return entry

    def seed(self, system, passages, result):
        """Import a complete, separately verified system result, never a subset.

        The caller must verify the imported source/model/runtime provenance.
        This method validates the matrix and primes exact-pair metric reuse.
        """
        rows = result.get("per_passage", [])
        by_id = {p["id"]: p for p in passages}
        if (result.get("system") != system or len(by_id) != len(passages)
                or len(rows) != len(passages) or {r["id"] for r in rows} != set(by_id)
                or any(not isinstance(r.get("output"), str)
                       or not all(isinstance(r.get(k), dict) for k in ("readability", "preservation"))
                       or r.get("domain", "") != by_id[r["id"]].get("domain", "")
                       for r in rows)):
            raise ValueError("Imported system result is incomplete or mismatched")
        for row in rows:
            passage = by_id[row["id"]]
            entry = {"identity": self._identity(passage, system),
                     **{k: row[k] for k in ("output", "readability", "preservation")}}
            existing = self._read(passage, system, "cells")
            if existing is not None and existing != entry:
                raise ValueError("Imported result conflicts with an existing cell")
            write_json(self._path(passage, system, "cells"), entry)
            pair = {"source": passage["original"], "output": row["output"]}
            path = self.root / "metrics" / (digest(list(pair.values())) + ".json")
            metrics = {"signature": self.signature, "pair": pair,
                       **{k: row[k] for k in ("readability", "preservation")}}
            if path.exists() and json.loads(path.read_text()) != metrics:
                raise ValueError("Imported metrics conflict for an identical pair")
            write_json(path, metrics)

    def system_result(self, system, passages):
        rows = []
        for passage in passages:
            entry = self._read(passage, system, "cells")
            if entry is None:
                raise ValueError(f"Missing matrix cell: {passage['id']} / {system}")
            rows.append({"id": passage["id"], "domain": passage.get("domain", ""),
                         **{k: entry[k] for k in ("output", "readability", "preservation")}})
        aggregate = {}
        for group in ("readability", "preservation"):
            for key in rows[0][group] if rows else []:
                values = [r[group][key] for r in rows if r[group][key] is not None]
                if values:
                    aggregate[f"{group}.{key}.mean"] = sum(values) / len(values)
                    aggregate[f"{group}.{key}.n"] = len(values)
        return {"system": system, "per_passage": rows, "aggregate": aggregate}
