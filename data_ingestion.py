import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion")
        try:
            # Load from notebook/data — user places cs-training.csv here
            data_path = os.path.join("notebook", "data", "cs-training.csv")
            df = pd.read_csv(data_path, index_col=0)
            logging.info(f"Dataset loaded: {df.shape}")

            # ── Basic cleaning ──────────────────────────────────────────
            # Rename target for clarity
            df.rename(
                columns={"SeriousDlqin2yrs": "loan_default"}, inplace=True
            )

            # Drop rows where target is null
            df.dropna(subset=["loan_default"], inplace=True)

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True,
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["loan_default"]
            )
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False
            )
            logging.info(
                f"Train: {train_set.shape}  Test: {test_set.shape}"
            )
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f"Train → {train_path}\nTest  → {test_path}")
