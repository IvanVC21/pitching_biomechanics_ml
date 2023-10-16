import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class DataHandler:
    """Handles the splitting and preprocessing of data."""

    def __init__(self, df):
        """
        Initialize the DataHandler.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def split_data(self):
        """
        Split the data into training, validation, and test sets and apply Min-Max scaling.

        Returns:
        - Tuple: Tuple containing x_train, x_val, x_test, y_train, y_val, y_test.
        """
        x = self.df.iloc[:, 1:]
        y = self.df['pitch_speed_mph']

        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1/3, random_state=42)

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def merge_validation_and_test_sets(self, x_val, y_val, x_test, y_test):
        """
        Merge validation and test sets.

        Parameters:
        - x_val (np.ndarray): Validation features.
        - y_val (np.ndarray): Validation values.
        - x_test (np.ndarray): Test features.
        - y_test (np.ndarray): Test values.

        Returns:
        - Tuple: Tuple containing merged x_test and y_test.
        """
        x_test = np.concatenate((x_val, x_test), axis=0)
        y_test = np.concatenate((y_val, y_test), axis=0)
        return x_test, y_test


class HyperparameterOptimizer:
    """Optimizes hyperparameters using Optuna."""

    def __init__(self, n_startup_trials, n_trials):
        """
        Initialize the HyperparameterOptimizer.

        Parameters:
        - n_startup_trials (int): Number of initial trials for TPESampler.
        - n_trials (int): Total number of trials for optimization.
        """
        self.n_startup_trials = n_startup_trials
        self.n_trials = n_trials

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

        study = optuna.create_study(direction='minimize', study_name='XGBoost Regression Optimization', 
                                    sampler=TPESampler(n_startup_trials=self.n_startup_trials))
        study.optimize(objective, n_trials=self.n_trials)

        print('Best hyperparameters: %s', study.best_params)
        return study.best_params


class XGBoostTrainer:
    """Trains an XGBoost model and saves predictions."""

    def train_and_evaluate_xgboost(self, x_train, y_train, x_test, y_test, best_params):
        """
        Train and evaluate an XGBoost model.

        """
        opt_xgb = xgboost.XGBRegressor(**best_params)
        opt_xgb.fit(x_train, y_train)
        y_pred = opt_xgb.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print('Optimized XGBoost Regressor R2 on Test Data: %.2f' % round(r2, 2))
        print('Optimized XGBoost Regressor RMSE on Test Data: %.2f' % round(rmse, 2))

        return opt_xgb, y_pred

    def save_model_and_predictions(self, model, y_pred, x_train, y_train, x_test, y_test, df):
        """
        Save the trained model and predictions for the test set.

        """
        results_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'results')
        os.makedirs(results_directory, exist_ok=True)

        model_save_path = os.path.join(results_directory, 'pitching_speed_prediction_model.pkl')
        with open(model_save_path, 'wb') as file:
            pickle.dump(model, file)

        test_predictions = np.round(y_pred, 2)
        test_indices = range(len(y_train), len(y_train) + len(y_test))

        test_df = pd.DataFrame({
            'prediction': np.round(test_predictions, 2),
            'actual_speed': np.round(y_test, 2),
            'error': np.round(np.abs(test_predictions - y_test), 2)
        })

        test_df[df.columns[1:]] = df.iloc[test_indices, 1:].reset_index(drop=True)

        test_df.to_csv(os.path.join(results_directory, 'predictions.csv'), index=False)



class FeatureImportancePlotter:
    """Plots feature importance based on PCA."""

    def plot_feature_importance(self, x_train, x_test, df):
        pca = PCA()
        X_train_pca = pca.fit_transform(x_train)
        X_test_pca = pca.transform(x_test)

        pca_components = np.abs(pca.components_[:2, :])
        feature_importance = np.sum(pca_components, axis=0)

        feature_importance_df = pd.DataFrame({
            'Feature': df.columns[1:],
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

        results_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'results')
        os.makedirs(results_directory, exist_ok=True)

        plt.savefig(os.path.join(results_directory, 'feature_importance.png'))









