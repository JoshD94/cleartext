"""Structure v2: sentence operations beyond splitting (opt-in, after F2).

Operations: passive to active (agentive, and "we" recovered from context),
appositive / non-restrictive relative extraction, fronted subordinate
clauses (split with a connective, or moved after the main clause), empty
"there is ... who" / "it is ... that" frames, and light verb +
nominalization to a verb. See models/docs/STRUCTURE_V2.md.
"""

from .stage import KEPT, KEPT_V3, OPS, StructureV2

__all__ = ["KEPT", "KEPT_V3", "OPS", "StructureV2"]
