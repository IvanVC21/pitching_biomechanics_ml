# train_and_evaluate.py
import numpy as np
import pandas as pd
import xgboost
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data
import os
import pickle
import logging

def train_and_evaluate_xgboost(df, model_save_path='xgboost_model.pkl',
                                predictions_save_path='predictions.csv', 
                                feature_importance_plot_path='feature_importance.png'):
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    try:
        x = df.iloc[:, 1:]
        y = df['pitch_speed_mph']

        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1/3, random_state=42)

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        def objective(trial):
            param = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.025, 1),
                'gamma': trial.suggest_float('gamma', 0.00, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.05, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 6),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 6),
                'subsample': trial.suggest_float('subsample', 0.05, 1),
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000)
            }
            model = xgboost.XGBRegressor(**param)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            return np.sqrt(mean_squared_error(y_val, y_pred))

        study = optuna.create_study(direction='minimize', study_name='XGBoost Regression Optimization', sampler=TPESampler(n_startup_trials=200))
        study.optimize(objective, n_trials=2000)

        print('Best hyperparameters:', study.best_params)

        x_test = np.concatenate((x_val, x_test), axis=0)
        y_test = np.concatenate((y_val, y_test), axis=0)

        opt_xgb = xgboost.XGBRegressor(**study.best_params)
        opt_xgb.fit(x_train, y_train)
        y_pred = opt_xgb.predict(x_test)

        # Print R2 and RMSE on test data
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print('Optimized XGBoost Regressor RMSE on Test Data:', rmse)
        print('Optimized XGBoost Regressor R2 on Test Data:', r2)
        

        # Feature importance based on PCA
        pca = PCA()
        X_train_pca = pca.fit_transform(x_train)
        X_test_pca = pca.transform(x_test)

        pca_components = np.abs(pca.components_[:2, :])
        feature_importance = np.sum(pca_components, axis=0)

        feature_importance_df = pd.DataFrame({
            'Feature': df.columns[1:],
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        # Plot the most important features
        top_features_df = feature_importance_df.head(15).sort_values(by='Importance', ascending=True)
        plt.figure(figsize=(12, 8))  # Increase the figure height to provide more space for labels
        plt.barh(top_features_df['Feature'], top_features_df['Importance'], color='dodgerblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Top 15 Important Features Based on PCA')
        plt.tight_layout()  # Adjust layout for better spacing
        plt.grid(True)


        # Save the trained model using joblib
        script_directory = os.path.dirname(os.path.abspath(__file__))
        feature_importance_plot_path = os.path.join(script_directory, feature_importance_plot_path)
        plt.savefig(feature_importance_plot_path)

        # Save the trained model using pickle
        with open(model_save_path, 'wb') as file:
            pickle.dump(opt_xgb, file)

        # Add prediction, actual speed, and error columns to the dataset
        df['Prediction'] = np.concatenate((opt_xgb.predict(x_train), y_pred))
        df['Actual_Speed'] = np.concatenate((y_train, y_test))
        df['Error'] = np.abs(df['Prediction'] - df['Actual_Speed'])

        # Reorder columns
        df = df[['Prediction', 'Actual_Speed', 'Error'] + df.columns.tolist()[1:]]

        # Remove the final three columns
        df = df.iloc[:, :-3]

        # Save the dataset without the final three columns as a CSV named 'predictions.csv' in the same directory
        predictions_save_path = os.path.join(script_directory, 'predictions.csv')
        df.to_csv(predictions_save_path, index=False)

        return opt_xgb, df

    except Exception as e:
        # Log the error
        logging.error(f"Error in XGBoost training and evaluation: {str(e)}")

        # Raise the error to be caught by the calling code
        raise




