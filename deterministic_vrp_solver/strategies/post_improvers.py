from typing import Dict, List


class CrossExchangeImprover:
    def __init__(self, max_iters: int = 1):
        self.max_iters = int(max_iters)

    def improve(self, route_optimizer, routes, assignment, polygons_df, provider, time_cap: int):
        try:
            return route_optimizer.improve_routes_cross_exchange(
                routes, assignment, polygons_df, time_cap=time_cap, max_iters=self.max_iters
            )
        except Exception:
            return routes


class ALNSImprover:
    def __init__(self, max_iters: int = 1, remove_k: int = 2):
        self.max_iters = int(max_iters)
        self.remove_k = int(remove_k)

    def improve(self, route_optimizer, routes, assignment, polygons_df, provider, time_cap: int):
        try:
            from ..alns import improve_routes_alns
            return improve_routes_alns(
                routes, assignment, polygons_df, route_optimizer, max_iters=self.max_iters, remove_k=self.remove_k, time_cap=time_cap
            )
        except Exception:
            return routes


class PortalRefinementImprover:
    def improve(self, route_optimizer, routes, assignment, polygons_df, provider, time_cap: int):
        try:
            from ..portal_refinement import refine_routes_portals
            return refine_routes_portals(routes, provider, warehouse_id=getattr(route_optimizer, 'warehouse_id', 0))
        except Exception:
            return routes


class CompositeImprover:
    def __init__(self, improvers: List):
        self.improvers = list(improvers)

    def improve(self, route_optimizer, routes, assignment, polygons_df, provider, time_cap: int):
        current = routes
        for imp in self.improvers:
            current = imp.improve(route_optimizer, current, assignment, polygons_df, provider, time_cap)
        return current


