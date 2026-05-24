import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    try:
        logging.info("=" * 60)
        logging.info("TRAINING PIPELINE STARTED")
        logging.info("=" * 60)

        # Step 1 — Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Step 2 — Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        # Step 3 — Model Training
        model_trainer = ModelTrainer()
        best_model_name, best_model_info, model_report = (
            model_trainer.initiate_model_trainer(train_arr, test_arr)
        )

        logging.info("=" * 60)
        logging.info(f"TRAINING COMPLETE — Best Model: {best_model_name}")
        logging.info(f"ROC-AUC : {best_model_info['roc_auc']}")
        logging.info(f"Accuracy: {best_model_info['accuracy']}")
        logging.info(f"F1 Score: {best_model_info['f1_score']}")
        logging.info("=" * 60)

        return best_model_name, best_model_info, model_report

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    best_name, best_info, report = run_training_pipeline()
    print(f"\n✅ Best Model : {best_name}")
    print(f"   ROC-AUC   : {best_info['roc_auc']}")
    print(f"   Accuracy  : {best_info['accuracy']}")
    print(f"   F1 Score  : {best_info['f1_score']}")
    print(f"   Precision : {best_info['precision']}")
    print(f"   Recall    : {best_info['recall']}")
    print("\n📊 All Models:")
    for name, info in report.items():
        print(
            f"   {name:22s} | ROC-AUC: {info['roc_auc']} | Acc: {info['accuracy']} | F1: {info['f1_score']}"
        )
