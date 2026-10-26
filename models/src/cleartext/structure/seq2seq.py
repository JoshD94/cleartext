"""Bounded local T5-small fine-tuning, with dev-only checkpoint selection."""

import gc
import json
import math
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .corpus import write_json

PREFIX = "split sentence: "


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
        pressure = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if pressure.stdout.strip() != "1":
            raise RuntimeError(
                "Stopping own experiment: Mac memory pressure is not normal"
            )
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
):
    import torch
    from huggingface_hub import model_info
    from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer

    run = Path(run)
    capacity_check(run)
    random.seed(4701)
    np.random.seed(4701)
    torch.manual_seed(4701)
    torch.set_num_threads(4)
    device = choose_device(device)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(0.45)
    revision = model_info(model_name, token=False).sha
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, token=False
    )
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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, revision=revision, token=False
    ).to(device)
    model.config.use_cache = False
    learning_rate = 1e-3 if optimizer_name == "adafactor" else 3e-4
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
        "model": model_name,
        "revision": revision,
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
        "length_filtered_training_pairs": len(data["train"]) - len(train),
        "dev_pairs": len(dev),
        "selection": "Lowest token-weighted dev teacher-forced loss; test not used",
        "prefix": PREFIX,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    write_json(run / "seq2seq_config.json", configuration)
    start, last_check, step, best, history = time.monotonic(), 0.0, 0, math.inf, []

    def validate(current_model):
        current_model.eval()
        total, tokens = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(dev), batch_size):
                batch = _batch(tokenizer, dev[i : i + batch_size], device)
                count = int((batch["labels"] != -100).sum().item())
                total += float(current_model(**batch).loss.item()) * count
                tokens += count
        return total / max(1, tokens)

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
        "training_examples_seen": (epoch * len(train))
        + min(len(train), offset + effective_batch),
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
        checks = structure_checks(source, raw, doc)
        checks["checks"]["generation_terminated"] = terminated
        if not terminated:
            checks["failed"].append("generation_terminated")
            checks["pass"] = False
        return {
            "raw": raw,
            "output": raw if checks["pass"] else source,
            "raw_checks": checks,
        }
