"""Manifest contract tests using tiny files and the standard library only."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "scripts/experiments"
sys.path.insert(0, str(EXPERIMENTS))
spec = importlib.util.spec_from_file_location("coverage_compare", EXPERIMENTS / "candidate_coverage_compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "models"
        self.args = SimpleNamespace(
            protocol=self.root / "protocol.json", audit=self.root / "audit",
            guards=self.root / "guards", guard_source=self.root / "guard.py",
            baseline=self.root / "baseline", base_run=self.root / "run",
            output=self.root / "output", encoder_snapshot=self.root / "bert-snapshot",
            nltk_data=self.root / "nltk_data", freeze_manifest=None,
        )
        self.source = self.put("src/cleartext/member.py", "# frozen source\n")
        self.put("guard.py", "# corrected guard\n")
        self.put("protocol.json", {
            "variant": "unique_dictionary_verb_roundtrip_v1", "thresholds": [.45, .80],
            "meaning_floor": .10, "seeds": [0, 1, 2], "folds": 10,
        })
        self.put("audit/resources/infl_lu.csv.gz", "dictionary bytes")
        self.put("audit/records.jsonl", "audit rows")
        self.put("audit/manifest.json", {
            "input_sha256": self.hashes("audit/resources/infl_lu.csv.gz"),
            "source_sha256": self.hashes("src/cleartext/member.py"),
        })
        self.put("audit/summary.json", {
            "manifest_sha256": compare.digest(self.root / "audit/manifest.json"),
            "records_sha256": compare.digest(self.root / "audit/records.jsonl"),
        })
        candidates = ("data/cache/refine-v7-benchls-dev.pkl", "data/cache/refine-v7-benchls-holdout.pkl")
        bert = "outputs/bert-candidate-coverage-20261002/records.jsonl"
        for name in (*candidates, bert, "baseline/feature-tables.npz", "baseline/feature-keys.json",
                     *(f"baseline/similarity-seed{seed}.npz" for seed in range(3))):
            self.put(name, name)
        candidate_hashes = {name: compare.digest(self.root / name) for name in candidates}
        self.put("baseline/summary.json", {
            "feature_tables_sha256": compare.digest(self.root / "baseline/feature-tables.npz"),
            "input_sha256": {**candidate_hashes, bert: compare.digest(self.root / bert)},
        })
        self.put("guards/records.jsonl", "guard rows")
        self.put("guards/manifest.json", {
            "source_sha256": {"aligned": compare.digest(self.args.guard_source)},
            "input_sha256": candidate_hashes,
        })
        self.put("guards/summary.json", {"records_sha256": compare.digest(self.root / "guards/records.jsonl")})
        self.put("run/config.json", {
            "decision_features": "prompted_similarity", "sense_model": "run/sense_model.pkl",
            "checker": "run/checker.pkl", "decision": "run/decision.pkl",
        })
        for name in ("run/sense_model.pkl", "run/checker.pkl", "run/decision.pkl", "run/stacker.pkl",
                     "runs/initial-20260924/word_model.pkl", "runs/initial-20260924/context_model.pkl",
                     "runs/context-20260924/phrase_model.pkl", "data/cache/brown-bigrams.pkl",
                     "bert-snapshot/config.json", "bert-snapshot/tokenizer_config.json",
                     "bert-snapshot/vocab.txt", "bert-snapshot/tokenizer.json", "bert-snapshot/model.safetensors",
                     "nltk_data/corpora/wordnet.zip", "nltk_data/corpora/cmudict/cmudict",
                     "nltk_data/corpora/stopwords/english", "packages/en_core_web_sm/parser/model",
                     "packages/en_core_web_md/vocab/vectors", "packages/lemminflect/resources/infl_lu.csv.gz",
                     "packages/lemminflect/resources/model_infl.pkl.gz", "packages/lemminflect/resources/model_lemma.pkl.gz",
                     "packages/wordfreq/data/large_en.msgpack.gz", "packages/wordfreq/data/small_en.msgpack.gz"):
            self.put(name, name)
        for target, value in (("ROOT", self.root), ("source_paths", lambda: [self.source]),
                              ("package_directory", lambda name: self.root / "packages" / name),
                              ("runtime_versions", lambda: {"python": "test runtime"})):
            mock = patch.object(compare, target, value)
            mock.start()
            self.addCleanup(mock.stop)

    def put(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return path

    def hashes(self, *names):
        return {str(self.root / name): compare.digest(self.root / name) for name in names}

    def manifest(self):
        protocol, audit, inputs = compare.preflight(self.args)
        return compare.make_manifest(self.args, protocol, audit, inputs)

    def main(self, *extra):
        argv = ["compare"]
        for name in ("protocol", "audit", "guards", "guard_source", "baseline", "base_run", "output",
                     "encoder_snapshot", "nltk_data"):
            argv += ["--" + name.replace("_", "-"), str(getattr(self.args, name))]
        with patch.object(sys, "argv", argv + list(extra)), contextlib.redirect_stdout(io.StringIO()) as output:
            compare.main()
        return output.getvalue()

    def test_preflight_requires_implicit_word_context_and_phrase_models(self):
        inputs = compare.preflight(self.args)[2]
        for name in ("runs/initial-20260924/word_model.pkl", "runs/initial-20260924/context_model.pkl",
                     "runs/context-20260924/phrase_model.pkl"):
            with self.subTest(name=name):
                path = self.root / name
                self.assertEqual(inputs[str(path)], compare.digest(path))
                original = path.read_bytes()
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    compare.preflight(self.args)
                path.write_bytes(original)

    def test_changed_implicit_model_bytes_invalidate_freeze_and_pinned_reproduction(self):
        for name in ("word_model.pkl", "context_model.pkl"):
            with self.subTest(name=name):
                manifest = self.manifest()
                pinned = self.put("pinned.json", manifest)
                self.args.freeze_manifest = pinned
                self.assertEqual(self.manifest(), manifest)
                path = self.root / "runs/initial-20260924" / name
                original = path.read_bytes()
                # Equal size rules out implementations that only freeze file sizes.
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                with self.assertRaisesRegex(ValueError, name):
                    compare.validate_freeze(manifest)
                with self.assertRaisesRegex(ValueError, "pinned manifest differs: input_sha256"):
                    self.manifest()
                self.args.freeze_manifest = None
                self.assertNotEqual(self.manifest()["input_sha256"][str(path)], manifest["input_sha256"][str(path)])
                path.write_bytes(original)

    def test_resources_include_native_dependencies_and_reject_changes(self):
        manifest = self.manifest()
        resources = manifest["resource_sha256"]
        for name in ("bert-snapshot/tokenizer.json", "bert-snapshot/model.safetensors",
                     "packages/en_core_web_sm/parser/model", "packages/en_core_web_md/vocab/vectors",
                     "packages/lemminflect/resources/model_infl.pkl.gz",
                     "packages/lemminflect/resources/model_lemma.pkl.gz", "packages/wordfreq/data/large_en.msgpack.gz",
                     "data/cache/brown-bigrams.pkl", "nltk_data/corpora/cmudict/cmudict",
                     "nltk_data/corpora/stopwords/english", "nltk_data/corpora/wordnet.zip"):
            self.assertIn(str(self.root / name), resources)
        compare.validate_freeze(manifest)
        self.put("bert-snapshot/added_tokens.json", "{}")
        with self.assertRaisesRegex(ValueError, "resource file inventory"):
            compare.validate_freeze(manifest)

    def test_cache_symlink_retargeting_invalidates_freeze(self):
        weights = self.args.encoder_snapshot / "model.safetensors"
        weights.unlink()
        first = self.put("blobs/first", "first weights")
        second = self.put("blobs/second", "other weights")
        weights.symlink_to(first)
        manifest = self.manifest()
        self.assertIn(str(weights), manifest["resource_sha256"])
        weights.unlink()
        weights.symlink_to(second)
        with self.assertRaisesRegex(ValueError, "resource_sha256.*model.safetensors"):
            compare.validate_freeze(manifest)

    def test_prior_manifest_pins_snapshot_and_rejects_old_incomplete_manifest(self):
        manifest = self.manifest()
        self.args.freeze_manifest = self.put("pinned.json", manifest)
        self.args.encoder_snapshot = None
        with patch.object(compare, "encoder_snapshot", side_effect=AssertionError("must not resolve moving main")):
            self.assertEqual(self.manifest(), manifest)
        self.put("pinned.json", {"input_sha256": manifest["input_sha256"]})
        with self.assertRaisesRegex(ValueError, "pinned manifest differs: freeze_version"):
            self.manifest()

    def test_post_native_drift_prevents_success_summary(self):
        for name in ("runs/initial-20260924/word_model.pkl", "runs/initial-20260924/context_model.pkl",
                     "src/cleartext/member.py", "bert-snapshot/tokenizer.json"):
            with self.subTest(name=name):
                self.args.output = self.root / ("output-" + name.replace("/", "_"))
                path = self.root / name
                original = path.read_bytes()

                def drift(*args):
                    path.write_bytes(original + b"changed during native scoring")
                    return {}

                with patch.object(compare, "native_experiment", side_effect=drift) as native:
                    with self.assertRaisesRegex(ValueError, "frozen .* changed"):
                        self.main()
                    native.assert_called_once()
                self.assertFalse((self.args.output / "summary.json").exists())
                path.write_bytes(original)

    def test_success_validates_freeze_and_preflight_never_calls_native(self):
        pinned = self.root / "preflight-freeze.json"
        with patch.object(compare, "native_experiment", return_value={}) as native:
            self.assertEqual(json.loads(self.main("--preflight", "--write-freeze-manifest", str(pinned)))["preflight"], "passed")
            self.assertEqual(json.loads(pinned.read_text()), self.manifest())
            self.main("--preflight", "--freeze-manifest", str(pinned))
            native.assert_not_called()
            self.assertFalse(self.args.output.exists())
            self.main()
            native.assert_called_once()
        summary = json.loads((self.args.output / "summary.json").read_text())
        self.assertTrue(summary["freeze_validated_after_native"])
        self.assertFalse(summary["historical_model_bytes_verified"])


if __name__ == "__main__":
    unittest.main()
