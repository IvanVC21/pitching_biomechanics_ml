from train_and_evaluate import train_and_evaluate_xgboost
from preprocess_data import preprocess_data
import os


if __name__ == "__main__":
    mech_csv = 'poi_metrics.csv'
    meta_csv = 'metadata.csv'
    
    # Preprocess data
    df = preprocess_data(mech_csv, meta_csv)

    # Train and evaluate the XGBoost model
    trained_model, result_df = train_and_evaluate_xgboost(df)