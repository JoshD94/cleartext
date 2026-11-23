"""Structure v2: sentence operations beyond splitting, applied after F2.

    from cleartext.structure_v2 import StructureV2
    stage = StructureV2(ops=("passive", "appositive"))
    text, edits = stage.rewrite(f2_output)

Each operation rewrites one sentence and is checked on its own: the F2
hard checks (numbers, negation, entities, technical phrases, code literals)
must pass, and an optional entailment scorer (`entailment(a, b) -> [0, 1]`,
for example teammate_stages.EntailmentVeto().score, the F2 split floor) must
reach `min_entailment` in both directions' minimum. A failing edit is
dropped and the sentence keeps its previous text. Operations run in the
order given; each sees the previous one's output. Nothing here is on by
default; F2 is unchanged.
"""

from .appositive import AppositiveSplit
from .common import hard_checks, parse
from .frames import EmptyFrames
from .fronted import FrontedClause
from .nominal import Nominalization
from .passive import ActiveVoice

OPS = {
    "passive": lambda: ActiveVoice(),
    "passive_we": lambda: ActiveVoice(agents=("we",)),
    "passive_v3": lambda: ActiveVoice(version=3),
    "passive_v4": lambda: ActiveVoice(version=4),
    # Structure v3 (round 6): v4 plus a whitelist, a second-parser agreement
    # check, an output parse check and a BERT fluency floor; by-phrase
    # passives only. Round 6 adds the 0.5 entailment floor (min_entailment).
    "passive_v5": lambda: ActiveVoice(version=5, kinds=("agentive",)),
    "passive_recovered": lambda: ActiveVoice(agents=("we",), kinds=("recovered_we",)),
    "appositive": lambda: AppositiveSplit(),
    "appositive_v2": lambda: AppositiveSplit(version=2),
    "appositive_v3": lambda: AppositiveSplit(version=3),
    "appositive_v4": lambda: AppositiveSplit(version=4),  # round 6: definite or named anchors
    "fronted_split": lambda: FrontedClause(rules=("split",)),
    "fronted_move": lambda: FrontedClause(rules=("move",)),
    "frames": lambda: EmptyFrames(),
    # Round 2: existential "there" only. Round 1 showed "It is X that ..." is
    # usually a referential "it" ("It is a highway that links ..."), not a cleft.
    "frames_v2": lambda: EmptyFrames(rules=("existential",)),
    "nominal": lambda: Nominalization(),
    "nominal_v2": lambda: Nominalization(version=2),
}
CONTEXT = 3  # preceding sentences an operation may read (agent recovery)
# The operations that met the pre-registered bar on fresh text (majority
# Good >= 85%, Changes meaning <= 10%; evals/reports/20261122-structure-v2),
# in the combined stage's order. Opt-in: StructureV2(ops=KEPT).
KEPT = ("passive_recovered", "fronted_split")
# Structure v3 (rounds 6-7): the stricter bar (Good >= 90%, meaning <= 8%)
# was met by passive_v5 and appositive_v4, rated with the 0.5 entailment
# floor: StructureV2(ops=KEPT_V3, entailment=EntailmentVeto().score,
# min_entailment=0.5). Order as in the dev run (DEV_V3.md).
KEPT_V3 = ("passive_v5", "passive_recovered", "appositive_v4", "fronted_split")


def build(name, **kwargs):
    if name not in OPS:
        raise ValueError(f"Unknown structure v2 operation {name}; known: {sorted(OPS)}")
    return OPS[name]() if not kwargs else type(OPS[name]())(**kwargs)


class StructureV2:
    def __init__(self, ops=("passive",), entailment=None, min_entailment=None, checks=True, operators=None):
        self.names = tuple(ops)
        self.ops = list(operators) if operators is not None else [build(n) for n in self.names]
        if min_entailment is not None and entailment is None:
            raise ValueError("min_entailment needs an entailment scorer")
        self.entailment, self.min_entailment, self.checks = entailment, min_entailment, checks

    def accept(self, before, after):
        """(accepted, reason) for one sentence edit."""
        if self.checks:
            failed = hard_checks(before, after)
            if failed:
                return False, "hard checks: " + ",".join(failed)
        if self.min_entailment is not None:
            score = self.entailment(before, after)
            if score < self.min_entailment:
                return False, f"entailment {score:.3f}"
        return True, None

    def rewrite_sentence(self, sentence, context=()):
        """(text, accepted edits, rejected edits) for one sentence."""
        current, accepted, rejected = sentence, [], []
        for op in self.ops:
            out_parts = []
            # An earlier operation may have made two sentences; later ones see each.
            parts = [s.text for s in parse(current).sents]
            changed = False
            for part in parts:
                new, edits = op.rewrite_sentence(part, context)
                if new != part:
                    ok, reason = self.accept(part, new)
                    if ok:
                        accepted += [e.as_dict() for e in edits]
                        out_parts.append(new)
                        changed = True
                        continue
                    rejected += [{**e.as_dict(), "reason": reason} for e in edits]
                out_parts.append(part)
            if changed:
                current = " ".join(out_parts)
        return current, accepted, rejected

    def rewrite(self, text):
        """(text, edits) for a passage, the shape EnsembleClearText stages use."""
        doc = parse(text)
        spans = [(s.start_char, s.end_char) for s in doc.sents]
        parsed = list(doc.sents)
        parts, edits, offset = [], [], 0
        for k, (start, end) in enumerate(spans):
            context = [p.as_doc() for p in parsed[max(0, k - CONTEXT): k]]
            new, accepted, _ = self.rewrite_sentence(text[start:end], context)
            parts += [text[offset:start], new]
            offset = end
            edits += accepted
        parts.append(text[offset:])
        return "".join(parts), edits
