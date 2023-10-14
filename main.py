from preprocess_data import DataPreprocessor
from train_and_evaluate import XGBoostTrainer

if __name__ == "__main__":
    mech_csv = 'poi_metrics.csv'
    meta_csv = 'metadata.csv'

    # Preprocess data
    preprocessor = DataPreprocessor(mech_csv, meta_csv)
    df = preprocessor.preprocess()

    # Train and evaluate the XGBoost model
    xgboost_trainer = XGBoostTrainer(df)
    trained_model, result_df = xgboost_trainer.train_and_evaluate()