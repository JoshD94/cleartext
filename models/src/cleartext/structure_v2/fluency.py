"""BERT-base pseudo-log-likelihood (structure v3 fluency floor).

`pll(text)` masks each word piece in turn and sums the log-probabilities of
the true pieces under the pipeline's own `bert-base-uncased`
(cleartext.contextual). A rewrite that keeps the source's words but scores
far below the source ("the national UK teachers will close ... schools
strike", "Provider Z. owns them") is usually garbled.
"""

from functools import lru_cache


@lru_cache(maxsize=4096)
def pll(text, max_pieces=126):
    import torch
    from ..contextual import model
    tok, mlm = model()
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_pieces + 2)["input_ids"][0]
    n = len(ids) - 2
    if n <= 0:
        return 0.0
    batch = ids.repeat(n, 1)
    pos = torch.arange(1, n + 1)
    batch[torch.arange(n), pos] = tok.mask_token_id
    with torch.no_grad():
        logits = mlm(batch).logits
    lp = torch.log_softmax(logits[torch.arange(n), pos], -1)[torch.arange(n), ids[1:n + 1]]
    return float(lp.sum())


def fluency_drop(before, after):
    """pll(after) - pll(before): 0 for an equally fluent rewrite, very
    negative for a garbled one."""
    return pll(after) - pll(before)
