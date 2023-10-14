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
import os
import pickle
import logging

class XGBoostTrainer:
    def __init__(self, df):
        self.df = df
        self.logger = logging.getLogger(__name__)
        self._logger_setup()

    def _logger_setup(self):
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def split_data(self):
        x = self.df.iloc[:, 1:]
        y = self.df['pitch_speed_mph']

        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1/3, random_state=42)

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def optimize_hyperparameters(self, x_train, y_train, x_val, y_val):
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

        print('Best hyperparameters: %s', study.best_params)
        return study.best_params

    def merge_validation_and_test_sets(self, x_val, y_val, x_test, y_test):
        x_test = np.concatenate((x_val, x_test), axis=0)
        y_test = np.concatenate((y_val, y_test), axis=0)
        return x_test, y_test

    def train_and_evaluate_xgboost(self, x_train, y_train, x_test, y_test):
        opt_xgb = xgboost.XGBRegressor(**self.optimize_hyperparameters(x_train, y_train, x_test, y_test))
        opt_xgb.fit(x_train, y_train)
        y_pred = opt_xgb.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print('Optimized XGBoost Regressor R2 on Test Data: %.2f' % round(r2, 2))
        print('Optimized XGBoost Regressor RMSE on Test Data: %.2f' % round(rmse, 2))

        return opt_xgb, y_pred

    def plot_feature_importance(self, x_train, x_test):
        pca = PCA()
        X_train_pca = pca.fit_transform(x_train)
        X_test_pca = pca.transform(x_test)

        pca_components = np.abs(pca.components_[:2, :])
        feature_importance = np.sum(pca_components, axis=0)

        feature_importance_df = pd.DataFrame({
            'Feature': self.df.columns[1:],
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        top_features_df = feature_importance_df.head(15).sort_values(by='Importance', ascending=True)
        plt.figure(figsize=(12, 8))
        plt.barh(top_features_df['Feature'], top_features_df['Importance'], color='dodgerblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Top 15 Important Features Based on PCA')
        plt.tight_layout()
        plt.grid(True)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        results_directory = os.path.join(script_directory, 'results')
        os.makedirs(results_directory, exist_ok=True)

        feature_importance_plot_path = os.path.join(results_directory, 'feature_importance.png')
        plt.savefig(feature_importance_plot_path)

    def save_model_and_predictions(self, model, y_pred, x_train, y_train, x_test, y_test):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        results_directory = os.path.join(script_directory, 'results')
        os.makedirs(results_directory, exist_ok=True)

        model_save_path = os.path.join(results_directory, 'pitching_speed_prediction_model.pkl')
        with open(model_save_path, 'wb') as file:
            pickle.dump(model, file)

        train_predictions = model.predict(x_train)
        test_predictions = y_pred

        self.df['Prediction'] = np.concatenate((train_predictions, test_predictions))
        self.df['Actual_Speed'] = np.concatenate((y_train, y_test))
        self.df['Error'] = np.abs(self.df['Prediction'] - self.df['Actual_Speed'])

        self.df = self.df[['Prediction', 'Actual_Speed', 'Error'] + self.df.columns.tolist()[1:-3]]

        predictions_save_path = os.path.join(results_directory, 'predictions.csv')
        self.df.to_csv(predictions_save_path, index=False)

    def train_and_evaluate(self):
        try:
            x_train, x_val, x_test, y_train, y_val, y_test = self.split_data()
            x_test, y_test = self.merge_validation_and_test_sets(x_val, y_val, x_test, y_test)
            opt_xgb, y_pred = self.train_and_evaluate_xgboost(x_train, y_train, x_test, y_test)
            self.plot_feature_importance(x_train, x_test)
            self.save_model_and_predictions(opt_xgb, y_pred, x_train, y_train, x_test, y_test)

            return opt_xgb, self.df
        except Exception as e:
            self.logger.error("Error in XGBoost training and evaluation: %s", str(e))
            raise









