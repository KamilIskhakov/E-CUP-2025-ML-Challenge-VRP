from pathlib import Path
import polars as pl


def load_orders_lazy(orders_path: Path) -> pl.LazyFrame:
    df = pl.read_json(str(orders_path))
    lf = df.explode("Orders").unnest("Orders").lazy()
    return lf
