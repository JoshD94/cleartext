# ClearText evaluation results

Generated: 2026-11-02T20:11:25.499173+00:00 | Passages: 18

| system | readability.flesch_reading_ease.mean | readability.flesch_kincaid_grade.mean | readability.gunning_fog.mean | readability.avg_sentence_length.mean | readability.technical_term_count.mean | preservation.tfidf_cosine_similarity.mean | preservation.embedding_similarity.mean | preservation.nli_forward.mean | preservation.nli_backward.mean | preservation.entities_preserved.mean | preservation.negations_preserved.mean | preservation.tenses_preserved.mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0.15 | 19.17 | 23.55 | 23.87 | 3.00 | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 |
| rule_based | 10.33 | 17.64 | 21.31 | 23.43 | 2.50 | 0.86 | 0.97 | 0.99 | 0.99 | 0.94 | 1.00 | 1.00 |
| lexical_only | 4.06 | 18.62 | 22.78 | 23.87 | 2.39 | 0.93 | 0.98 | 0.95 | 0.71 | 1.00 | 1.00 | 1.00 |
| phrase_only | -1.85 | 19.38 | 23.74 | 23.59 | 3.00 | 0.97 | 1.00 | 0.99 | 0.99 | 0.94 | 1.00 | 1.00 |
| structure_only | 0.15 | 19.17 | 23.55 | 23.87 | 3.00 | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 |
| lexical_structural | 4.06 | 18.62 | 22.78 | 23.87 | 2.39 | 0.93 | 0.98 | 0.95 | 0.71 | 1.00 | 1.00 | 1.00 |
| full_guarded | 2.20 | 18.81 | 22.95 | 23.59 | 2.39 | 0.91 | 0.98 | 0.96 | 0.72 | 0.94 | 1.00 | 1.00 |
| full_no_guardrails | 2.20 | 18.81 | 22.95 | 23.59 | 2.39 | 0.91 | 0.98 | 0.96 | 0.72 | 0.94 | 1.00 | 1.00 |
