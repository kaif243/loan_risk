import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


FEATURE_COLUMNS = [
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


class PredictPipeline:
    def __init__(self):
        self.model       = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            features_transformed = self.preprocessor.transform(features)
            pred  = self.model.predict(features_transformed)
            proba = (
                self.model.predict_proba(features_transformed)[:, 1]
                if hasattr(self.model, "predict_proba")
                else np.zeros(len(pred))
            )
            return pred, proba
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Maps Streamlit form inputs → DataFrame row for the pipeline."""

    def __init__(
        self,
        revolving_utilization: float,
        age: int,
        past_due_30_59: int,
        debt_ratio: float,
        monthly_income: float,
        open_credit_lines: int,
        times_90_days_late: int,
        real_estate_loans: int,
        past_due_60_89: int,
        dependents: int,
    ):
        self.revolving_utilization = revolving_utilization
        self.age                   = age
        self.past_due_30_59        = past_due_30_59
        self.debt_ratio            = debt_ratio
        self.monthly_income        = monthly_income
        self.open_credit_lines     = open_credit_lines
        self.times_90_days_late    = times_90_days_late
        self.real_estate_loans     = real_estate_loans
        self.past_due_60_89        = past_due_60_89
        self.dependents            = dependents

    def get_data_as_dataframe(self) -> pd.DataFrame:
        data = {
            "RevolvingUtilizationOfUnsecuredLines": [self.revolving_utilization],
            "age":                                  [self.age],
            "NumberOfTime30-59DaysPastDueNotWorse": [self.past_due_30_59],
            "DebtRatio":                            [self.debt_ratio],
            "MonthlyIncome":                        [self.monthly_income],
            "NumberOfOpenCreditLinesAndLoans":       [self.open_credit_lines],
            "NumberOfTimes90DaysLate":              [self.times_90_days_late],
            "NumberRealEstateLoansOrLines":         [self.real_estate_loans],
            "NumberOfTime60-89DaysPastDueNotWorse": [self.past_due_60_89],
            "NumberOfDependents":                   [self.dependents],
        }
        return pd.DataFrame(data)
