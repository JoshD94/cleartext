"""BERT-base as a context reader: slot probabilities for candidates and contextual word vectors.

Loaded lazily and shared. Used by two ensemble members: BertSense (meaning) and BertSlotFit (fit).
CPU only; four threads.
"""

from functools import lru_cache
import numpy as np

NAME = "bert-base-uncased"


@lru_cache(1)
def model():
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import transformers

    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(NAME)
    mlm = AutoModelForMaskedLM.from_pretrained(NAME).eval()
    return tok, mlm


def slot_scores(context, start, end, words, max_pieces=6, window=150, limit=320):
    """Mean log-probability of each word's pieces in context[start:end], one piece masked at a time.
    Inputs longer than `limit` pieces keep `window` pieces on each side of the slot (same rule as the cached
    SWORDS scores from scripts/experiments/bert_benchmark.py)."""
    import torch

    tok, mlm = model()
    before = tok(context[:start], add_special_tokens=False).input_ids
    after = tok(context[end:], add_special_tokens=False).input_ids
    batch = []
    owner = []
    for i, w in enumerate(words):
        pieces = tok((" " if start else "") + w, add_special_tokens=False).input_ids[
            :max_pieces
        ]
        ids = [tok.cls_token_id] + before + pieces + after + [tok.sep_token_id]
        pos = 1 + len(before)
        if len(ids) > limit:
            cut = max(0, len(before) - window)
            ids = (
                [tok.cls_token_id]
                + before[cut:]
                + pieces
                + after[:window]
                + [tok.sep_token_id]
            )
            pos = 1 + len(before) - cut
        for j, p in enumerate(pieces):
            x = list(ids)
            x[pos + j] = tok.mask_token_id
            batch.append((x, pos + j, p))
            owner.append(i)
    out = np.zeros(len(words))
    n = np.zeros(len(words))
    with torch.no_grad():
        for k in range(0, len(batch), 32):
            chunk = batch[k : k + 32]
            L = max(len(x) for x, _, _ in chunk)
            ids = torch.tensor(
                [x + [tok.pad_token_id] * (L - len(x)) for x, _, _ in chunk]
            )
            att = (ids != tok.pad_token_id).long()
            lp = mlm(input_ids=ids, attention_mask=att).logits.log_softmax(-1)
            for r, (x, p, piece) in enumerate(chunk):
                out[owner[k + r]] += float(lp[r, p, piece])
                n[owner[k + r]] += 1
    return out / np.maximum(n, 1)


def word_vectors(sentences):
    """For [(words, [indices])]: unit vectors (mean of the last four layers over each word's pieces), one list per sentence."""
    import torch

    tok, mlm = model()
    out = []
    with torch.no_grad():
        for k in range(0, len(sentences), 16):
            chunk = sentences[k : k + 16]
            enc = tok(
                [w for w, _ in chunk],
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            h = torch.stack(
                mlm.bert(**enc, output_hidden_states=True).hidden_states[-4:]
            ).mean(0)
            for r, (w, idx) in enumerate(chunk):
                ids = enc.word_ids(r)
                vs = []
                for i in idx:
                    pos = [j for j, x in enumerate(ids) if x == i]
                    v = (
                        h[r, pos].mean(0).numpy()
                        if pos
                        else np.zeros(h.shape[-1], dtype=np.float32)
                    )
                    vs.append(v / (np.linalg.norm(v) or 1))
                out.append(vs)
    return out


def word_vector(words, index):
    return word_vectors([(list(words), [index])])[0][0]
