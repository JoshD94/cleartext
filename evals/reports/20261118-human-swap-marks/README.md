# Human marks on 60 word swaps (2026-11-18)

Joshua marked the 60 swaps from the NLI-calibration review sample (`../20261117-nli-calibration`) blind, with NLI scores hidden. The marks were Good, Changes meaning, or Not simpler. Only word pairs, bands, scores and marks are kept here (`marks.json`); the sentences stay local.

## Result

| NLI band | Share of all swaps | Good | Changes meaning | Not simpler |
|---|---|---|---|---|
| ≥0.95 | 74% | 8/15 | 5/15 | 2/15 |
| 0.90–0.95 | 6% | 2/15 | 8/15 | 5/15 |
| 0.80–0.90 | 4% | 2/15 | 11/15 | 2/15 |
| <0.80 | 16% | 1/15 | 12/15 | 2/15 |
| **All 60** | | **13** | **36** | **11** |

- Weighted by band share, about **42% of the swaps the word stage proposes are good**.
- **The NLI gate is not too strict.**
  - Lowering the threshold only adds bad swaps: at 0.80, 12 of 45 accepted are good (27%).
  - At 0.95, 8 of 15 accepted are good (53%), and 5 of those 15 still change meaning.
  - The 0.95 bound is close to the best available, and it is still too lenient.
- **Nouns are the worst.**
  - 1 of 24 noun swaps was good. Technical terms are replaced with wrong words: packets→messages, nonce→present, latency→potential, metabolite→substance.
  - Verbs: 7/24 good. Adjectives: 4/10 good.
  - Restricting to non-nouns at ≥0.95 still gives only 7/12 good.

## Conclusion

The word stage is the problem, not the verification. No threshold gives an acceptable error rate. Recommendation: keep words off (F2), stop tuning the word gate, and handle hard vocabulary by explaining it rather than replacing it.

Part 2 of the page (60 F2 sentence splits) is not marked yet.
