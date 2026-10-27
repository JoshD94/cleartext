# Sentence simplification bottleneck audit

The audit separates model copying, rejected rewrites, target/check conflicts and
candidate selection. It also reproduces selected-checkpoint training exposure
and runs teacher-forced copy/reference comparisons. All evaluation inputs had
already been inspected. This is development diagnosis, not a new test gain.

Read the [findings and next steps](../../docs/STRUCTURE_BOTTLENECKS.md).

- [Aggregate results](summary.json)
- [Frozen inputs, checkpoint and source hashes](freeze.json)
- [Initial protocol](protocol.json) and [bounded follow-up](protocol-v2.json)
- [Authored probes](probes.json) and [their check results](authored_probes.json)
- [Environment](environment.json), [resources](resources.json) and [validation](validation.json)

The later 128-example sample has 76 raw copies, 33 rejected proposed changes and
19 returned changes. Of 97 changed references, the model copies 46 inputs and
returns changes on 19. Prepared training contains 366 changed targets that fail
the current checks. This count mixes unsuitable targets and false rejections;
it is not a count of proven annotation errors.

No retraining, runtime gate changes or demo promotion occurred. Full reference
failures, every checked beam and per-example losses remain outside Git under
`~/.local/state/cleartext/structure-audit-20261027/run-v2/`. The earlier run is
preserved in `run/`. The eight authored expectations are not human ratings or
training gold. The reserved domain test remains unused.
