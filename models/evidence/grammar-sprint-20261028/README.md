# Local grammar and benchmark integrity checks

The private LanguageTool 6.6 trial retained all 96 existing development inputs and all 17 separate authored diagnostics. Both correction conditions reproduced all 96 archived input corrections and produced identical new preprocessing outputs. Three requests timed out, two in the one-pass condition and one in follow-up. Their unchanged outputs stay in the denominator and are marked unavailable. These failures prevent treating the panel as a complete successful checker run. No model inference followed.

All 17 diagnostics completed in both conditions. Both retain the 14 existing mechanism expectations. One report-only input changes between conditions: He dont know the result. becomes He don't know the result. under one pass and He doesn't know the result. with the bounded follow-up. This is a native rule-mechanism result, not general grammar or semantic accuracy.

The independent frozen passage audit passes all 1900 saved predictions across 190 inputs and 10 expected systems. The scorer now enforces that expected matrix, including shared omissions and missing systems, before metrics.

Yoga ran the private server on 8788 in cleartext-grammar-sprint-20261028, 3 GiB cap, CPU 100%, after placement permitted. Earlier 8787 was occupied by an unrelated service and was preserved. A second pre-data attempt lacked Java on PATH. The successful attempt used the existing JDK 17 executable by absolute path. The private server exited afterward. Raw responses and complete logs remain outside Git in ~/.local/state/cleartext/improvement-sprint-20261028/grammar-pipeline/. No weights, running demo, benchmark membership or correctness labels changed.
