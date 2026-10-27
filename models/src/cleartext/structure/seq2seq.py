"""Bounded local T5-small fine-tuning, with dev-only checkpoint selection."""

import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .corpus import write_json

PREFIX = "split sentence: "


def complement_scopes(doc):
    """Keep embedded claims under their original predicate, including conjuncts.

    Splitting "said that revenue rose and costs fell" must retain attribution for
    both claims. Lemma/dependency counts are conservative parser diagnostics;
    they do not resolve speakers, coreference or every kind of semantic scope.
    Only forward complements are tracked. Comma splices can parse as backward
    complements; treating those as attribution would reject useful repairs.
    """
    result = Counter()
    for token in doc:
        if token.pos_ not in {"VERB", "AUX", "ADJ"}:
            continue
        current, seen = token, set()
        while current.dep_ == "conj" and current.i not in seen:
            seen.add(current.i)
            current = current.head
        if current.dep_ in {"ccomp", "xcomp"} and current.i > current.head.i:
            result[
                (current.dep_, current.head.lemma_.lower(), token.lemma_.lower())
            ] += 1
    return result


def predicate_tenses(doc):
    """Source-relative tense signatures, allowing a repeated shared auxiliary.

    Sets allow "was prepared and approved" to repeat "was" after a split.
    They do not prove that a tense is attached to the correct participant.
    """
    result = set()
    for token in doc:
        if token.pos_ not in {"VERB", "AUX", "ADJ"}:
            continue
        current, seen = token, set()
        while current.i not in seen:
            seen.add(current.i)
            finite = [
                p
                for p in [
                    current,
                    *[
                        c
                        for c in current.children
                        if c.dep_ in {"aux", "auxpass", "cop"}
                    ],
                ]
                if p.pos_ in {"VERB", "AUX"} and "Fin" in p.morph.get("VerbForm")
            ]
            if finite:
                result.update(
                    (
                        token.lemma_.lower(),
                        tuple(p.morph.get("Tense")),
                        tuple(p.morph.get("Mood")),
                    )
                    for p in finite
                )
                break
            if current.dep_ != "conj":
                break
            current = current.head
    return result


def choose_device(requested="auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def capacity_check(run):
    if shutil.disk_usage(run).free < 2 * 1024**3:
        raise RuntimeError("Stopping own experiment: less than 2 GiB disk free")
    if platform.system() == "Darwin":
        while True:
            pressure = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            if pressure == "1":
                write_json(Path(run) / "capacity_status.json", {"state": "running"})
                break
            if pressure != "2" or os.environ.get("CLEARTEXT_WAIT_FOR_CAPACITY") != "1":
                raise RuntimeError(
                    "Stopping own experiment: Mac memory pressure is not normal"
                )
            write_json(
                Path(run) / "capacity_status.json",
                {
                    "state": "paused",
                    "reason": "Mac warning pressure; training state retained",
                },
            )
            print(
                "Training paused for Mac memory pressure; rechecking in 30 seconds",
                flush=True,
            )
            time.sleep(30)
            if shutil.disk_usage(run).free < 2 * 1024**3:
                raise RuntimeError("Stopping own experiment: less than 2 GiB disk free")
        result = subprocess.run(
            ["memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # -Q emits a percentage rather than inducing memory pressure.
        import re

        match = re.search(r"free percentage:\s*(\d+)%", result.stdout)
        if match and int(match[1]) < 10:
            raise RuntimeError(
                "Stopping own experiment: Mac free memory percentage below 10"
            )
    elif Path("/proc/meminfo").exists():
        info = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        if int(info["MemAvailable"].split()[0]) < 2 * 1024**2:
            raise RuntimeError("Stopping own experiment: less than 2 GiB available RAM")


def _batch(tokenizer, rows, device):
    inputs = tokenizer(
        [PREFIX + r["source"] for r in rows], padding=True, return_tensors="pt"
    )
    targets = tokenizer(
        text_target=[r["reference"] for r in rows], padding=True, return_tensors="pt"
    )
    if inputs["input_ids"].shape[1] > 192 or targets["input_ids"].shape[1] > 256:
        raise ValueError(
            "Pair exceeds experiment token limit; refusing silent truncation"
        )
    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return {**{k: v.to(device) for k, v in inputs.items()}, "labels": labels.to(device)}


def train_seq2seq(
    data,
    run,
    max_steps=400,
    batch_size=4,
    device="auto",
    model_name="google-t5/t5-small",
    optimizer_name="adafactor",
    accumulation_steps=1,
    initial_checkpoint=None,
    target_learning_rate=None,
):
    import torch
    from huggingface_hub import model_info
    from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer

    run = Path(run)
    if (run / "seq2seq").exists() and any((run / "seq2seq").iterdir()):
        raise FileExistsError("Refusing to overwrite an existing seq2seq checkpoint")
    if target_learning_rate is not None and (
        not math.isfinite(target_learning_rate) or target_learning_rate <= 0
    ):
        raise ValueError("Learning rate must be finite and positive")
    checkpoint = None
    if initial_checkpoint is not None:
        initial_checkpoint = Path(initial_checkpoint).resolve()
        destination = (run / "seq2seq").resolve()
        if initial_checkpoint.is_relative_to(destination) or destination.is_relative_to(
            initial_checkpoint
        ):
            raise ValueError(
                "Initial checkpoint and output must be separate directories"
            )
        if not (initial_checkpoint / "config.json").is_file() or not any(
            (initial_checkpoint / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        ):
            raise ValueError("Initial checkpoint needs config and local model weights")
        hashes = {}
        for path in sorted(initial_checkpoint.iterdir()):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[path.name] = digest.hexdigest()
        checkpoint = {"path": str(initial_checkpoint), "files_sha256": hashes}
    capacity_check(run)
    random.seed(4701)
    np.random.seed(4701)
    torch.manual_seed(4701)
    torch.set_num_threads(4)
    device = choose_device(device)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(0.45)
    if checkpoint is None:
        revision = model_info(model_name, token=False).sha
        load_name = model_name
        load_options = {"revision": revision, "token": False}
    else:
        revision = None
        load_name = str(initial_checkpoint)
        load_options = {"local_files_only": True, "token": False}
    tokenizer = AutoTokenizer.from_pretrained(load_name, **load_options)
    # Filter training only on length. No tokens are truncated at inference either.
    train = []
    for row in data["train"]:
        if (
            len(tokenizer(PREFIX + row["source"])["input_ids"]) <= 192
            and len(tokenizer(text_target=row["reference"])["input_ids"]) <= 256
        ):
            train.append(row)
    dev = [
        r
        for r in data["dev"]
        if len(tokenizer(PREFIX + r["source"])["input_ids"]) <= 192
        and len(tokenizer(text_target=r["reference"])["input_ids"]) <= 256
    ]
    if not train or not dev:
        raise ValueError("No token-length eligible training/development pairs")
    model = AutoModelForSeq2SeqLM.from_pretrained(load_name, **load_options).to(device)
    model.config.use_cache = False
    learning_rate = (
        target_learning_rate
        if target_learning_rate is not None
        else (1e-3 if optimizer_name == "adafactor" else 3e-4)
    )
    warmup_steps = max(1, max_steps // 10)
    if optimizer_name == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            clip_threshold=1.0,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    effective_batch = batch_size * accumulation_steps
    epochs = max(
        1, math.ceil(max_steps / max(1, math.ceil(len(train) / effective_batch)))
    )
    configuration = {
        "model": load_name,
        "revision": revision,
        "initial_checkpoint": checkpoint,
        "optimizer_state": "New optimizer; this is weight initialization, not optimizer/step resumption",
        "device": str(device),
        "seed": 4701,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch,
        "max_source_tokens": 192,
        "max_target_tokens": 256,
        "training_pairs": len(train),
        "training_domains": dict(Counter(r.get("domain", "general") for r in train)),
        "length_filtered_training_pairs": len(data["train"]) - len(train),
        "dev_pairs": len(dev),
        "selection": "Lowest token-weighted dev teacher-forced loss, including step 0; test not used",
        "prefix": PREFIX,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    write_json(run / "seq2seq_config.json", configuration)
    start, last_check, step, best, history = time.monotonic(), 0.0, 0, math.inf, []
    examples_seen = 0

    def validate(current_model):
        current_model.eval()
        total, tokens = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(dev), batch_size):
                batch = _batch(tokenizer, dev[i : i + batch_size], device)
                count = int((batch["labels"] != -100).sum().item())
                total += float(current_model(**batch).loss.item()) * count
                tokens += count
        result = total / max(1, tokens)
        if not math.isfinite(result):
            raise RuntimeError("Non-finite development loss")
        return result

    # Retain initialization when continuation makes development loss worse.
    # Previously the first trained checkpoint could replace a better initializer.
    best = validate(model)
    history.append({"step": 0, "dev_loss": best})
    model.save_pretrained(run / "seq2seq", safe_serialization=True)
    tokenizer.save_pretrained(run / "seq2seq")
    write_json(run / "seq2seq" / "selection.json", {"step": 0, "dev_loss": best})
    print("dev", 0, best, flush=True)

    for epoch in range(epochs):
        order = list(train)
        random.Random(4701 + epoch).shuffle(order)
        for offset in range(0, len(order), effective_batch):
            if step >= max_steps:
                break
            if time.monotonic() - last_check > 90:
                capacity_check(run)
                last_check = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            group = order[offset : offset + effective_batch]
            reported_loss = 0.0
            for micro_offset in range(0, len(group), batch_size):
                micro = group[micro_offset : micro_offset + batch_size]
                batch = _batch(tokenizer, micro, device)
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                weight = len(micro) / len(group)
                (loss * weight).backward()
                reported_loss += float(loss.item()) * weight
            if optimizer_name == "adamw":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate * min(
                    1.0, (step + 1) / warmup_steps
                )
            optimizer.step()
            step += 1
            examples_seen += len(group)
            if step == 1 or step % 20 == 0:
                elapsed = time.monotonic() - start
                progress = {
                    "stage": "train-seq2seq",
                    "step": step,
                    "max_steps": max_steps,
                    "loss": reported_loss,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": elapsed / step * (max_steps - step),
                    "device": str(device),
                    "host": platform.node(),
                }
                write_json(run / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if (
                step % 100 == 0
                or step == max_steps
                or offset + effective_batch >= len(order)
            ):
                dev_loss = validate(model)
                history.append({"step": step, "dev_loss": dev_loss})
                print("dev", step, dev_loss, flush=True)
                if dev_loss < best:
                    best = dev_loss
                    model.save_pretrained(run / "seq2seq", safe_serialization=True)
                    tokenizer.save_pretrained(run / "seq2seq")
                    write_json(
                        run / "seq2seq" / "selection.json",
                        {"step": step, "dev_loss": best},
                    )
        if step >= max_steps:
            break
    result = {
        **configuration,
        "steps_completed": step,
        "training_examples_seen": examples_seen,
        "history": history,
        "selected": json.loads((run / "seq2seq" / "selection.json").read_text()),
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(run / "seq2seq_training.json", result)
    del model, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


class SentenceGenerator:
    def __init__(self, path, device="auto"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = choose_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
            .to(self.device)
            .eval()
        )
        self.model.config.use_cache = True

    def predict(self, source, doc=None):
        import torch

        from .common import structure_checks
        from ..features import nlp

        encoded = self.tokenizer(PREFIX + source, return_tensors="pt")
        if encoded["input_ids"].shape[1] > 192:
            return {
                "raw": source,
                "output": source,
                "reason": "source exceeds token limit",
                "raw_checks": structure_checks(source, source, doc),
            }
        with torch.inference_mode():
            ids = self.model.generate(
                **{k: v.to(self.device) for k, v in encoded.items()},
                num_beams=2,
                do_sample=False,
                max_new_tokens=256,
                no_repeat_ngram_size=0,
            )
        raw = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        terminated = self.tokenizer.eos_token_id in ids[0].tolist()[1:]
        old = doc if doc is not None else nlp()(source)
        new = nlp()(raw)
        checks = structure_checks(source, raw, old, new)
        same_tense = predicate_tenses(old) == predicate_tenses(new)
        checks["checks"]["predicate_tense"] = same_tense
        if not same_tense:
            checks["failed"].append("predicate_tense")
            checks["pass"] = False
        same_scope = complement_scopes(old) == complement_scopes(new)
        checks["checks"]["complement_scope"] = same_scope
        if not same_scope:
            checks["failed"].append("complement_scope")
            checks["pass"] = False
        checks["checks"]["generation_terminated"] = terminated
        if not terminated:
            checks["failed"].append("generation_terminated")
            checks["pass"] = False
        return {
            "raw": raw,
            "output": raw if checks["pass"] else source,
            "raw_checks": checks,
        }
