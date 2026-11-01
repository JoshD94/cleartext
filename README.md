# cleartext

The trained simplification models are under [`models/`](models/README.md).
The [combined-pipeline benchmark](models/docs/PIPELINE_BENCHMARK.md) connects the
sentence and word models to the teammate evaluator, with stage comparisons and
separate precision/coverage results.
The [November 1 status-report replay](evals/reports/20261101-status-rerun/README.md)
compares the submitted six-passage results with explicit trained-stack settings
and reruns the larger frozen benchmark.

## Week 2 evaluation

See [`evaluation/README.md`](./evaluation/README.md) for the evaluator,
usage, implementation status, temporary baseline components, and remaining
work.
