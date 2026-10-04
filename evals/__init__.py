"""ClearText evaluation package.

Implements the evaluation system from Section 3 of the ClearText project
proposal: readability/simplicity metrics, meaning-preservation metrics,
module-level detection and ranking metrics, baseline systems, and the
ablation runner.
"""

from . import baselines, detection, preservation, readability, runner

__all__ = ["baselines", "detection", "preservation", "readability", "runner"]
