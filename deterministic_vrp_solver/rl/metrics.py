import logging


logger = logging.getLogger(__name__)


class MetricsRecorder:
    def __init__(self):
        self.rows = []

    def append(self, **kwargs):
        self.rows.append(kwargs)

    def save(self, parquet_path: str, csv_path: str):
        try:
            import polars as pl
            df = pl.DataFrame(self.rows)
            df.write_parquet(parquet_path, compression="zstd")
            df.write_csv(csv_path)
            logger.info(f"Метрики обучения сохранены: {parquet_path}, {csv_path}")
        except Exception as e:
            logger.error(f"Не удалось сохранить метрики обучения: {e}")


