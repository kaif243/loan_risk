import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_report_file_path: str  = os.path.join("artifacts", "model_report.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train/test arrays")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test  = test_array[:, :-1]
            y_test  = test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    max_depth=6, random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            }

            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models
            )

            # ── Pick best model by ROC-AUC ──────────────────────────────
            best_model_name = max(
                model_report, key=lambda k: model_report[k]["roc_auc"]
            )
            best_model_info = model_report[best_model_name]
            best_model      = best_model_info["model"]

            if best_model_info["roc_auc"] < 0.60:
                raise CustomException("No model met the minimum ROC-AUC threshold.", sys)

            logging.info(
                f"Best model: {best_model_name}  ROC-AUC: {best_model_info['roc_auc']}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            save_object(
                file_path=self.model_trainer_config.model_report_file_path,
                obj=model_report,
            )

            return best_model_name, best_model_info, model_report

        except Exception as e:
            raise CustomException(e, sys)
