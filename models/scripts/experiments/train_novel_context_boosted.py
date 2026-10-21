"""One bounded boosted validator on the same saved SWORDS features and split."""

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from cleartext.audit import sha256
from cleartext.data import ROOT
from cleartext.novel_context import select_context_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    args = parser.parse_args()
    prior = ROOT / "outputs/novel-context-training-20261003"
    summary = json.loads((prior / "summary.json").read_text())
    arrays_path = prior / "training-arrays.npz"
    assert sha256(arrays_path) == summary["arrays_sha256"]
    arrays = np.load(arrays_path, allow_pickle=False)
    X, y, train, calibration = [
        arrays[name] for name in ("features", "labels", "train", "calibration")
    ]
    assert not set(arrays["groups"][train]) & set(arrays["groups"][calibration])
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("output and run must be new")
    sources = [Path(__file__).resolve(), ROOT / "src/cleartext/novel_context.py"]
    args.output.mkdir(parents=True, exist_ok=False)
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=10,
        random_state=4701,
    ).fit(X[train], y[train])
    probability = model.predict_proba(X[calibration])[:, 1]
    threshold, curve = select_context_threshold(y[calibration], probability)
    args.save_run.mkdir(parents=True, exist_ok=False)
    with (args.save_run / "validator.pkl").open("xb") as stream:
        pickle.dump(model, stream)
    config = json.loads(
        (ROOT / "runs/novel-context-validator-20261003/config.json").read_text()
    )
    config.update(
        context_threshold=threshold,
        estimator="hist_gradient_boosting",
        deployment_enabled=False,
    )
    (args.save_run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(
        args.output / "calibration.npz",
        probability=probability,
        labels=y[calibration],
        indices=calibration,
    )
    result = {
        key: summary[key]
        for key in (
            "scope",
            "label",
            "features",
            "pairs",
            "positive_pairs",
            "train_pairs",
            "calibration_pairs",
            "context_groups",
            "exclusions",
            "native_checks",
        )
    }
    result.update(
        estimator="hist_gradient_boosting",
        calibration_auc=float(roc_auc_score(y[calibration], probability)),
        calibration_log_loss=float(log_loss(y[calibration], probability)),
        context_threshold=threshold,
        threshold_curve=curve,
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in sources},
        input_sha256={
            str(path.relative_to(ROOT)): sha256(path)
            for path in (arrays_path, prior / "summary.json")
        },
        validator_sha256=sha256(args.save_run / "validator.pkl"),
        saved_run=str(args.save_run.resolve().relative_to(ROOT)),
        default_changed=False,
        new_labels=False,
        tsar_used=False,
        native_checks_reused_from=str(prior.relative_to(ROOT)),
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.save_run / "development.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "calibration_auc",
                    "calibration_log_loss",
                    "context_threshold",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
