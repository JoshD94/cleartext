| Config | Changed | FK ↓ | Flesch ↑ | Words/sent | NLI passage | NLI<0.8 | NLI sentence | Sent<0.8 | Entities | Num/neg fails | Gates | ASSET SARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| words=off phrases=authored structure=t5_unchecked negation=off grammar=off terms=off ◆ | 55/90 | 12.300 | 42.93 | 20.24 | 0.9571 | 4 | 0.9751 | 3 | 0.9853 | 1/0 | fail 3,4 |  |
| words=v7 phrases=simpleppdb structure=t5 negation=off grammar=off terms=off | 90/90 | 13.075 | 42.77 | 23.27 | 0.8424 | 21 | 0.8886 | 23 | 0.9494 | 0/1 | fail 1,2,3,4 |  |
| words=v7 phrases=authored+simpleppdb structure=t5 negation=off grammar=off terms=off | 90/90 | 13.076 | 42.75 | 23.26 | 0.8427 | 21 | 0.8889 | 23 | 0.9494 | 0/1 | fail 1,2,3,4 |  |
| words=off phrases=simpleppdb structure=t5+compound_tagger negation=off grammar=off terms=off | 73/90 | 13.128 | 41.87 | 22.98 | 0.9269 | 8 | 0.9452 | 9 | 0.9644 | 0/1 | fail 1,2,3,4 |  |
| words=off phrases=authored+simpleppdb structure=t5+compound_tagger negation=off grammar=off terms=off | 73/90 | 13.130 | 41.85 | 22.97 | 0.9275 | 8 | 0.9457 | 9 | 0.9644 | 0/1 | fail 1,2,3,4 |  |
| words=contextual phrases=simpleppdb structure=t5 negation=off grammar=off terms=off | 87/90 | 13.139 | 42.31 | 23.27 | 0.9027 | 10 | 0.9276 | 9 | 0.9522 | 0/1 | fail 1,2,3,4 |  |
| words=contextual phrases=authored+simpleppdb structure=t5 negation=off grammar=off terms=off | 87/90 | 13.140 | 42.29 | 23.26 | 0.9030 | 10 | 0.9280 | 9 | 0.9522 | 0/1 | fail 1,2,3,4 |  |
| words=off phrases=simpleppdb structure=t5 negation=off grammar=off terms=off | 72/90 | 13.238 | 41.60 | 23.27 | 0.9268 | 8 | 0.9451 | 9 | 0.9644 | 0/1 | fail 1,2,3,4 |  |
| words=off phrases=authored+simpleppdb structure=t5 negation=off grammar=off terms=off | 72/90 | 13.239 | 41.58 | 23.26 | 0.9274 | 8 | 0.9456 | 9 | 0.9644 | 0/1 | fail 1,2,3,4 |  |
| words=v7 phrases=authored structure=t5+compound_tagger negation=off grammar=off terms=off passive=on | 89/90 | 13.296 | 41.00 | 23.17 | 0.8457 | 24 | 0.9109 | 16 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=t5+compound_tagger negation=off grammar=off terms=off passive=agentive | 89/90 | 13.303 | 40.89 | 23.13 | 0.8332 | 20 | 0.9099 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=t5+compound_tagger negation=off grammar=off terms=off | 89/90 | 13.348 | 40.89 | 23.32 | 0.8399 | 19 | 0.9140 | 16 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=contextual_veto phrases=authored structure=t5+compound_tagger negation=off grammar=off terms=off | 84/90 | 13.394 | 40.57 | 23.32 | 0.9276 | 7 | 0.9549 | 5 | 0.9889 | 0/0 | fail 1,2,4 |  |
| words=contextual phrases=authored structure=t5+compound_tagger negation=off grammar=off terms=off | 84/90 | 13.407 | 40.48 | 23.32 | 0.9322 | 7 | 0.9587 | 4 | 0.9889 | 0/0 | fail 1,2,4 | 39.79 |
| words=v7 phrases=authored structure=t5 negation=off grammar=off terms=off passive=on | 89/90 | 13.408 | 40.73 | 23.47 | 0.8455 | 24 | 0.9109 | 16 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=t5 negation=off grammar=off terms=off passive=agentive | 89/90 | 13.415 | 40.62 | 23.43 | 0.8332 | 20 | 0.9099 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored+crf structure=t5 negation=off grammar=off terms=off | 89/90 | 13.432 | 40.64 | 23.51 | 0.8438 | 19 | 0.9144 | 16 | 0.9731 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=crf structure=t5 negation=off grammar=off terms=off | 89/90 | 13.433 | 40.66 | 23.53 | 0.8438 | 19 | 0.9144 | 16 | 0.9787 | 0/0 | fail 1,2,4 |  |
| words=off phrases=authored structure=t5+compound_tagger negation=off grammar=off terms=off passive=on ◆ | 45/90 | 13.448 | 39.92 | 23.17 | 0.9590 | 5 | 0.9703 | 3 | 1.0000 | 0/0 | fail 2 | 27.79 |
| words=off phrases=authored structure=t5_raw negation=off grammar=off terms=off ◆ | 19/90 | 13.453 | 39.93 | 23.20 | 0.9832 | 0 | 0.9871 | 0 | 1.0000 | 0/0 | pass | 24.58 |
| words=v7 phrases=authored structure=t5 negation=off grammar=off terms=off | 89/90 | 13.460 | 40.62 | 23.62 | 0.8399 | 19 | 0.9140 | 16 | 0.9861 | 0/0 | fail 1,2,4 | 40.54 |
| words=off phrases=authored structure=t5+compound_tagger negation=off grammar=off terms=off passive=agentive ◆ | 26/90 | 13.461 | 39.76 | 23.13 | 0.9841 | 0 | 0.9874 | 0 | 1.0000 | 0/0 | pass | 24.94 |
| words=off phrases=authored+crf structure=t5+compound_tagger negation=off grammar=off terms=off | 27/90 | 13.473 | 39.82 | 23.22 | 0.9837 | 0 | 0.9868 | 0 | 0.9870 | 0/0 | fail 4 | 25.19 |
| words=off phrases=crf structure=t5+compound_tagger negation=off grammar=off terms=off | 27/90 | 13.474 | 39.84 | 23.23 | 0.9837 | 0 | 0.9868 | 0 | 0.9926 | 0/0 | pass | 25.19 |
| words=contextual phrases=authored+crf structure=t5 negation=off grammar=off terms=off | 84/90 | 13.483 | 40.27 | 23.51 | 0.9323 | 7 | 0.9582 | 4 | 0.9759 | 0/0 | fail 1,2,4 | 39.80 |
| words=contextual phrases=crf structure=t5 negation=off grammar=off terms=off | 84/90 | 13.484 | 40.29 | 23.53 | 0.9323 | 7 | 0.9582 | 4 | 0.9815 | 0/0 | fail 1,2,4 | 39.80 |
| words=contextual_veto phrases=authored structure=t5 negation=off grammar=off terms=off | 84/90 | 13.506 | 40.30 | 23.62 | 0.9269 | 7 | 0.9548 | 5 | 0.9889 | 0/0 | fail 1,2,4 |  |
| words=off phrases=authored structure=t5+compound_tagger negation=off grammar=off terms=off | 20/90 | 13.506 | 39.77 | 23.32 | 0.9837 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass | 24.29 |
| words=off phrases=authored structure=t5+compound_tagger negation=on grammar=off terms=off | 20/90 | 13.506 | 39.77 | 23.32 | 0.9837 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass | 24.29 |
| words=off phrases=authored structure=t5+compound_tagger negation=off grammar=on terms=off | 20/90 | 13.506 | 39.77 | 23.32 | 0.9837 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass | 24.29 |
| words=off phrases=authored structure=t5+compound_tagger negation=off grammar=off terms=on | 20/90 | 13.506 | 39.77 | 23.32 | 0.9837 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass | 24.29 |
| words=off phrases=off structure=t5+compound_tagger negation=off grammar=off terms=off | 19/90 | 13.507 | 39.79 | 23.34 | 0.9837 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass | 24.19 |
| words=contextual phrases=authored structure=t5 negation=off grammar=off terms=off | 84/90 | 13.519 | 40.20 | 23.62 | 0.9320 | 7 | 0.9586 | 4 | 0.9889 | 0/0 | fail 1,2,4 | 39.81 |
| words=off phrases=authored structure=t5 negation=off grammar=off terms=off passive=on | 42/90 | 13.559 | 39.64 | 23.47 | 0.9589 | 5 | 0.9703 | 3 | 1.0000 | 0/0 | fail 2 |  |
| words=v7 phrases=simpleppdb structure=rules negation=off grammar=off terms=off | 89/90 | 13.571 | 41.54 | 24.58 | 0.8445 | 21 | 0.8927 | 21 | 0.9494 | 0/1 | fail 1,2,3,4 |  |
| words=off phrases=authored structure=t5 negation=off grammar=off terms=off passive=agentive | 22/90 | 13.573 | 39.48 | 23.43 | 0.9841 | 0 | 0.9874 | 0 | 1.0000 | 0/0 | pass |  |
| words=v7 phrases=authored+simpleppdb structure=rules negation=off grammar=off terms=off | 89/90 | 13.573 | 41.52 | 24.57 | 0.8448 | 21 | 0.8931 | 21 | 0.9494 | 0/1 | fail 1,2,3,4 |  |
| words=off phrases=authored+crf structure=t5 negation=off grammar=off terms=off | 24/90 | 13.584 | 39.55 | 23.51 | 0.9836 | 0 | 0.9868 | 0 | 0.9870 | 0/0 | fail 4 |  |
| words=off phrases=crf structure=t5 negation=off grammar=off terms=off | 24/90 | 13.585 | 39.57 | 23.53 | 0.9836 | 0 | 0.9868 | 0 | 0.9926 | 0/0 | pass |  |
| words=off phrases=authored structure=t5 negation=off grammar=off terms=off | 16/90 | 13.618 | 39.49 | 23.62 | 0.9836 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass | 24.23 |
| words=off phrases=authored structure=t5 negation=on grammar=off terms=off | 16/90 | 13.618 | 39.49 | 23.62 | 0.9836 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass |  |
| words=off phrases=authored structure=t5 negation=off grammar=on terms=off | 16/90 | 13.618 | 39.49 | 23.62 | 0.9836 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass |  |
| words=off phrases=authored structure=t5 negation=off grammar=off terms=on | 16/90 | 13.618 | 39.49 | 23.62 | 0.9836 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass |  |
| words=off phrases=off structure=t5 negation=off grammar=off terms=off | 15/90 | 13.618 | 39.52 | 23.63 | 0.9836 | 0 | 0.9873 | 0 | 1.0000 | 0/0 | pass |  |
| words=v7 phrases=authored structure=compound_tagger negation=off grammar=off terms=off passive=on | 87/90 | 13.692 | 40.10 | 24.26 | 0.8428 | 25 | 0.9074 | 19 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=compound_tagger negation=off grammar=off terms=off passive=agentive | 85/90 | 13.711 | 39.95 | 24.24 | 0.8307 | 21 | 0.9088 | 18 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=compound_tagger negation=off grammar=off terms=off | 85/90 | 13.764 | 39.93 | 24.45 | 0.8374 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=contextual phrases=authored structure=compound_tagger negation=off grammar=off terms=off | 83/90 | 13.822 | 39.51 | 24.45 | 0.9312 | 7 | 0.9591 | 4 | 0.9889 | 0/0 | fail 1,2,4 |  |
| words=off phrases=authored structure=compound_tagger negation=off grammar=off terms=off passive=on | 38/90 | 13.855 | 38.93 | 24.26 | 0.9581 | 5 | 0.9696 | 4 | 1.0000 | 0/0 | fail 2 |  |
| words=v7 phrases=authored structure=rules negation=off grammar=off terms=off passive=on | 87/90 | 13.875 | 39.66 | 24.75 | 0.8425 | 25 | 0.9074 | 19 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=off phrases=authored structure=compound_tagger negation=off grammar=off terms=off passive=agentive ◆ | 16/90 | 13.878 | 38.75 | 24.24 | 0.9865 | 0 | 0.9883 | 0 | 1.0000 | 0/0 | pass |  |
| words=v7 phrases=authored structure=rules negation=off grammar=off terms=off passive=agentive | 85/90 | 13.893 | 39.51 | 24.73 | 0.8306 | 21 | 0.9088 | 18 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored+crf structure=rules negation=off grammar=off terms=off | 85/90 | 13.913 | 39.54 | 24.83 | 0.8411 | 20 | 0.9133 | 17 | 0.9731 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=crf structure=rules negation=off grammar=off terms=off | 85/90 | 13.914 | 39.57 | 24.85 | 0.8411 | 20 | 0.9132 | 17 | 0.9787 | 0/0 | fail 1,2,4 |  |
| words=off phrases=authored structure=compound_tagger negation=off grammar=off terms=off | 7/90 | 13.930 | 38.74 | 24.45 | 0.9860 | 0 | 0.9881 | 0 | 1.0000 | 0/0 | pass | 21.34 |
| words=v7 phrases=authored structure=rules negation=off grammar=off terms=off | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 | 39.95 |
| words=v7 phrases=authored structure=off negation=off grammar=off terms=off | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=rules negation=on grammar=off terms=off | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=rules negation=off grammar=on terms=off | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=authored structure=rules negation=off grammar=off terms=on | 85/90 | 13.948 | 39.49 | 24.94 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=v7 phrases=off structure=rules negation=off grammar=off terms=off | 85/90 | 13.949 | 39.51 | 24.96 | 0.8373 | 20 | 0.9129 | 17 | 0.9861 | 0/0 | fail 1,2,4 |  |
| words=contextual phrases=authored structure=rules negation=off grammar=off terms=off | 83/90 | 14.007 | 39.07 | 24.94 | 0.9309 | 7 | 0.9590 | 4 | 0.9889 | 0/0 | fail 1,2,4 | 39.03 |
| words=off phrases=authored structure=rules negation=off grammar=off terms=off passive=on | 34/90 | 14.038 | 38.49 | 24.75 | 0.9578 | 5 | 0.9695 | 4 | 1.0000 | 0/0 | fail 2 |  |
| rule_based ◆ | 14/90 | 14.049 | 38.79 | 24.96 | 0.9867 | 0 | 0.9886 | 0 | 0.9907 | 0/0 | ref | 22.34 |
| words=off phrases=authored structure=rules negation=off grammar=off terms=off passive=agentive | 11/90 | 14.061 | 38.31 | 24.73 | 0.9863 | 0 | 0.9882 | 0 | 1.0000 | 0/0 | pass |  |
| words=off phrases=authored structure=rules negation=off grammar=off terms=off | 1/90 | 14.114 | 38.29 | 24.94 | 0.9859 | 0 | 0.9881 | 0 | 1.0000 | 0/0 | pass | 20.79 |
| words=off phrases=authored structure=off negation=off grammar=off terms=off | 1/90 | 14.114 | 38.29 | 24.94 | 0.9859 | 0 | 0.9881 | 0 | 1.0000 | 0/0 | pass |  |
| original | 0/90 | 14.115 | 38.32 | 24.96 | 0.9859 | 0 | 0.9881 | 0 | 1.0000 | 0/0 | ref | 20.45 |
| words=off phrases=off structure=off negation=off grammar=off terms=off | 0/90 | 14.115 | 38.32 | 24.96 | 0.9859 | 0 | 0.9881 | 0 | 1.0000 | 0/0 | pass |  |

◆ = on the FK/passage-NLI Pareto frontier.
