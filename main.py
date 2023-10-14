from preprocessing import DataPreprocessor
from training import XGBoostTrainer

if __name__ == "__main__":
    mech_csv = 'data/poi_metrics.csv'
    meta_csv = 'data/metadata.csv'

    # Preprocess data
    preprocessor = DataPreprocessor(mech_csv, meta_csv)
    df = preprocessor.preprocess()

    # Train and evaluate the XGBoost model
    xgboost_trainer = XGBoostTrainer(df)
    trained_model, result_df = xgboost_trainer.train_and_evaluate()