"""Opt-in per-swap meaning veto scored on a local window (November 18, 2026).

Pass an instance as `EnsembleClearText.analyze(word_veto=...)`. For each
candidate word swap the pipeline calls it with the sentence before and after
that one swap; it returns True to veto the swap, which is then skipped on its
own (with `veto_refill`, the next-ranked word may take the slot). Swaps that
pass are applied as before. Nothing here is on by default.

The veto finds the changed words, cuts the same window of `window` words on
each side from both sentences, and scores bidirectional entailment on the two
windows: a swap passes only if the mean of forward and backward entailment is
at least `threshold` and the explicit checks (numbers, negations, entities)
pass on the whole sentence. Scoring a window rather than the whole sentence
keeps a long sentence from diluting a good swap (see
evals/reports/20261118-swap-window-veto/).

The entailment and check functions are injected, so this module loads no
model: evals/swap_veto.py binds them to the evaluation NLI and the
evaluator's own checks.
"""


def changed_region(before, after):
    """(prefix, before_mid, after_mid, suffix): the shortest whole-word middle
    run in which the two texts differ, with the shared text around it."""
    n = min(len(before), len(after))
    p = 0
    while p < n and before[p] == after[p]:
        p += 1
    s = 0
    while s < n - p and before[len(before) - 1 - s] == after[len(after) - 1 - s]:
        s += 1
    # Widen to whitespace boundaries, so a partly shared word counts as changed.
    while p > 0 and not before[p - 1].isspace():
        p -= 1
    while s > 0 and not before[len(before) - s].isspace():
        s -= 1
    return before[:p], before[p:len(before) - s], after[p:len(after) - s], before[len(before) - s:]


def window_pair(before, after, window):
    """The two texts cut to `window` whitespace words on each side of the
    changed run (identical context on both sides); None keeps the full texts."""
    if window is None or before == after:
        return before, after
    prefix, mid_b, mid_a, suffix = changed_region(before, after)
    left = prefix.split()[-window:] if window > 0 else []
    right = suffix.split()[:window] if window > 0 else []

    def join(mid):
        return " ".join([*left, *mid.split(), *right])

    return join(mid_b), join(mid_a)


class WindowSwapVeto:
    """word_veto(sentence, edited) -> True to veto, from window NLI and checks.

    entail(premise, hypothesis) -> entailment probability (None = unscorable,
    which vetoes). checks(sentence, edited) -> True if numbers, negations and
    entities are kept. Each call appends a decision record to `log` when one
    is given (raw text: keep it local).
    """

    def __init__(self, entail, threshold, window=6, checks=None, log=None):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a probability")
        if window is not None and window < 0:
            raise ValueError("window must be >= 0 or None")
        self.entail, self.threshold, self.window = entail, threshold, window
        self.checks, self.log = checks, log

    def score(self, sentence, edited):
        before, after = window_pair(sentence, edited, self.window)
        forward, backward = self.entail(before, after), self.entail(after, before)
        mean = None if forward is None or backward is None else (forward + backward) / 2
        checks_ok = True if self.checks is None else bool(self.checks(sentence, edited))
        _, word, replacement, _ = changed_region(sentence, edited)
        return {"word": word.strip(), "replacement": replacement.strip(), "window_before": before,
                "window_after": after, "forward": forward, "backward": backward, "mean": mean,
                "checks": checks_ok,
                "passes": mean is not None and mean >= self.threshold and checks_ok}

    def __call__(self, sentence, edited):
        record = self.score(sentence, edited)
        if self.log is not None:
            self.log.append({"sentence": sentence, "edited": edited, **record})
        return not record["passes"]
