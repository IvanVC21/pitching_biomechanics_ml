from preprocessing import DataPreprocessor
from training import DataHandler, HyperparameterOptimizer, XGBoostTrainer, FeatureImportancePlotter


if __name__ == "__main__":
    mech_csv = 'data/poi_metrics.csv'
    meta_csv = 'data/metadata.csv'

    # Preprocess data
    preprocessor = DataPreprocessor(mech_csv, meta_csv)
    df = preprocessor.preprocess()

    # Data Handling
    data_handler = DataHandler(df)
    x_train, x_val, x_test, y_train, y_val, y_test = data_handler.split_data()
    x_test, y_test = data_handler.merge_validation_and_test_sets(x_val, y_val, x_test, y_test)

    # Hyperparameter Optimization
    hyperparameter_optimizer = HyperparameterOptimizer()
    best_params = hyperparameter_optimizer.optimize_hyperparameters(x_train, y_train, x_val, y_val)

    # XGBoost Model Training and Evaluation
    xgboost_model = XGBoostTrainer()
    trained_model, y_pred = xgboost_model.train_and_evaluate_xgboost(x_train, y_train, x_test, y_test, best_params)

    # Feature Importance Plotting
    feature_importance_plotter = FeatureImportancePlotter()
    feature_importance_plotter.plot_feature_importance(x_train, x_test, df)

    # Save Model and Predictions
    xgboost_model.save_model_and_predictions(trained_model, y_pred, x_train, y_train, x_test, y_test, df)