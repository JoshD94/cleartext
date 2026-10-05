# Complete BERT candidate scores

The original slot scorer silently truncates each replacement to six wordpieces before scoring it. Two phrases with the same first six pieces therefore receive identical scores even when their endings differ. The new opt-in scorer keeps the complete replacement and scores every piece.

For a long sentence, it keeps nearby context within a 320-piece input, including special tokens. It reduces surrounding context when necessary to fit the full candidate. Empty candidates, invalid offsets and candidates too long to fit raise explicit errors. Attention masks use sequence lengths, so a literal `[PAD]` token remains visible.

The scorer streams batches of 16 masked inputs. It runs BERT's vocabulary head only at the masked positions, avoiding the original full sequence-by-vocabulary allocation. This saves that allocation; it is not a measured end-to-end speed claim. Longer candidates still require more masked inputs.

Set `bert_slot_complete` to `true` in a separate configuration. The two fit members keep their existing names and feature positions, so the saved experimental run shares the prior fit and decision weights. No model is retrained. The original `contextual.py`, v7 configuration and demo remain unchanged.

## Native scoring check

In `They recorded the sequence carefully.`, the old scorer gives both `one two three four five six seven` and `one two three four five six banana` a score of -3.237707. It has discarded the distinguishing ending. The complete scorer gives them -2.693010 and -4.676178 respectively. These are ungraded boundary examples, not semantic training labels.

The complete scorer agrees with the original implementation when its truncation limit is raised to retain all pieces. Maximum error across the four checked candidates is 0.000000477. Short candidate scores also agree within that tolerance. A long-context geometry check retains all seven candidate pieces inside a 309-piece input.

Of the 14,617 cached BenchLS candidate rows, only `Brobdingnagianest` exceeds six pieces. It has seven and is proposed for `largest` in `benchls-232`. No target exceeds six pieces. This explains why ordinary benchmark scores can miss the truncation bug.

The paired development check rescored the affected case's BERT features and frozen fit probabilities. It keeps candidate order and all other features, trains only the baseline decision in each target-word fold, and predicts both variants at the existing threshold and meaning floor. All BenchLS is reused development. TSAR is not read.

Run from `models/`, using new paths and a resource-checked named Yoga tmux job:

```sh
.venv/bin/python scripts/experiments/check_complete_slot.py \
  --output outputs/complete-slot-another-check \
  --save-run runs/ensemble-complete-slot-another-run

.venv/bin/python scripts/experiments/check_complete_slot_development.py \
  --validation outputs/complete-slot-another-check \
  --output outputs/complete-slot-another-dev

.venv/bin/python scripts/audit_domains.py --baselines \
  --run runs/ensemble-complete-slot-another-run \
  --output outputs/domain-audit-complete-slot-another-run
```

The current experimental run is `runs/ensemble-complete-slot-20261002`. The native check is in `outputs/complete-slot-check-20261002/`, the development comparison in `outputs/complete-slot-dev-20261002/`, and the domain audit in `outputs/domain-audit-complete-slot-20261002/`. Each records source snapshots and hashes. Use the run explicitly; this change does not establish higher simplification accuracy or justify default promotion.

All three jobs completed successfully on Yoga in `cleartext-complete-slot-20261002`, capped at 3 GB and four threads. The development comparison changed zero selections in every repeat. Both variants scored net 269, 264 and 258, with pooled precision 71.5% and coverage 66.1%. All 13 domain diagnostic outputs match the prior windowed relation run. Its known detail-loss failures remain.

All 69 tests pass. Coverage includes complete candidates near the input limit, explicit failures, bounded batch ownership, literal padding-token attention and unchanged fit-member names. The native encoder comparison verifies numerical agreement separately from the mocked batching test. Fatal/undefined-name lint passes.
