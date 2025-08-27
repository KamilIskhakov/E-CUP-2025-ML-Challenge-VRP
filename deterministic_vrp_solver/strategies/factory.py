from typing import Optional

from .warm_start_strategy import WarmStartAssignment
from .rcsp_strategy import RCSPAssignment
from .column_generation_strategy import ColumnGenerationAssignment
from .post_improvers import CrossExchangeImprover, ALNSImprover, PortalRefinementImprover, CompositeImprover


def build_assignment_strategy(name: str):
    name = (name or "").strip().lower()
    if name in ("warm", "warm_start", "greedy"):
        return WarmStartAssignment()
    if name in ("rcsp",):
        return RCSPAssignment()
    if name in ("cg", "column_gen", "column_generation"):
        return ColumnGenerationAssignment()
                         
    return WarmStartAssignment()


def build_post_improver(chain: str):
    """chain example: "cross,alns,portals"""
    if not chain:
        return CompositeImprover([])
    parts = [p.strip().lower() for p in chain.split(",") if p.strip()]
    imps = []
    for p in parts:
        if p == "cross":
            imps.append(CrossExchangeImprover(max_iters=1))
        elif p == "alns":
            imps.append(ALNSImprover(max_iters=1, remove_k=2))
        elif p in ("portal", "portals", "portal_refine"):
            imps.append(PortalRefinementImprover())
    return CompositeImprover(imps)


