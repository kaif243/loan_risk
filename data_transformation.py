import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    # ── Feature columns (Give Me Some Credit dataset) ──────────────────
    NUMERICAL_FEATURES = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]
    TARGET = "loan_default"

    def get_data_transformer_object(self):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, self.NUMERICAL_FEATURES),
                ]
            )
            logging.info("Preprocessor pipeline created.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train & test data read successfully.")

            preprocessor = self.get_data_transformer_object()

            X_train = train_df[self.NUMERICAL_FEATURES]
            y_train = train_df[self.TARGET]
            X_test  = test_df[self.NUMERICAL_FEATURES]
            y_test  = test_df[self.TARGET]

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed  = preprocessor.transform(X_test)

            # ── Handle class imbalance with SMOTE ───────────────────────
            logging.info(
                f"Class distribution before SMOTE: {dict(pd.Series(y_train).value_counts())}"
            )
            smote = SMOTE(random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(
                X_train_transformed, y_train
            )
            logging.info(
                f"Class distribution after  SMOTE: {dict(pd.Series(y_train_bal).value_counts())}"
            )

            train_arr = np.c_[X_train_bal, np.array(y_train_bal)]
            test_arr  = np.c_[X_test_transformed, np.array(y_test)]

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info("Preprocessor saved.")

            return train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
