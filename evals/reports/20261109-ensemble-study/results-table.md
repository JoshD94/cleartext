| Config | Changed | FK ↓ | Flesch ↑ | Words/sent | NLI passage | NLI<0.8 | NLI sentence | Sent<0.8 | Entities | Num/neg fails | Gates | ASSET SARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| words=v7 phrases=authored structure=t5 negation=off grammar=off terms=off ◆ | 89/90 | 13.460 | 40.62 | 23.62 | 0.8399 | 19 | 0.9140 | 16 | 0.9861 | 0/0 | fail 1,2,4 | 40.54 |
| words=contextual phrases=authored structure=t5 negation=off grammar=off terms=off ◆ | 84/90 | 13.519 | 40.20 | 23.62 | 0.9320 | 7 | 0.9586 | 4 | 0.9889 | 0/0 | fail 1,2,4 | 39.81 |
| words=off phrases=authored structure=t5 negation=off grammar=off terms=off ◆ | 16/90 | 13.618 | 39.49 | 23.62 | 0.9836 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass | 24.23 |
| words=v7 phrases=authored structure=compound_tagger negation=off grammar=off terms=off | 85/90 | 13.764 | 39.93 | 24.45 | 0.8374 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=contextual phrases=authored structure=compound_tagger negation=off grammar=off terms=off | 83/90 | 13.822 | 39.51 | 24.45 | 0.9312 | 7 | 0.9591 | 4 | 0.9889 | 0/0 | fail 1,2,4 |  |
| words=off phrases=authored structure=compound_tagger negation=off grammar=off terms=off ◆ | 7/90 | 13.930 | 38.74 | 24.45 | 0.9860 | 0 | 0.9881 | 0 | 1.0000 | 0/0 | pass | 21.34 |
| words=v7 phrases=authored structure=rules negation=off grammar=off terms=off | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 | 39.95 |
| words=v7 phrases=authored structure=off negation=off grammar=off terms=off | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=rules negation=on grammar=off terms=off | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=rules negation=off grammar=on terms=off | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=rules negation=off grammar=off terms=on | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=off structure=rules negation=off grammar=off terms=off | 85/90 | 13.949 | 39.51 | 24.96 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=contextual phrases=authored structure=rules negation=off grammar=off terms=off | 83/90 | 14.007 | 39.07 | 24.94 | 0.9309 | 7 | 0.9590 | 4 | 0.9889 | 0/0 | fail 1,2,4 | 39.03 |
| rule_based ◆ | 14/90 | 14.049 | 38.79 | 24.96 | 0.9867 | 0 | 0.9886 | 0 | 0.9907 | 0/0 | ref | 22.34 |
| words=off phrases=authored structure=rules negation=off grammar=off terms=off | 1/90 | 14.114 | 38.29 | 24.94 | 0.9859 | 0 | 0.9881 | 0 | 1.0000 | 0/0 | pass | 20.79 |
| original | 0/90 | 14.115 | 38.32 | 24.96 | 0.9859 | 0 | 0.9881 | 0 | 1.0000 | 0/0 | ref | 20.45 |

◆ = on the FK/passage-NLI Pareto frontier.
