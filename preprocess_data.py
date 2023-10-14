import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

class DataPreprocessor:
    def __init__(self, mech_csv, meta_csv):
        self.mech_csv = mech_csv
        self.meta_csv = meta_csv

        # Configure logging only if it hasn't been configured yet
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(console_handler)

    def _read_data(self):
        mech = pd.read_csv(self.mech_csv)
        meta = pd.read_csv(self.meta_csv)
        return mech, meta

    def _merge_data(self, mech, meta):
        df = mech.join(meta[['session_height_m', 'session_mass_kg', 'age_yrs']])
        new_columns = ['session_height_m', 'session_mass_kg', 'age_yrs']

        for col in new_columns:
            df.insert(2, col, df.pop(col))

        column_to_move = df['pitch_speed_mph']
        df.drop(columns=['pitch_speed_mph', 'session_pitch', 'session', 'pitch_type', 'p_throws'], inplace=True)
        df.insert(0, 'pitch_speed_mph', column_to_move)
        return df

    def preprocess(self):
        try:
            mech, meta = self._read_data()
            df = self._merge_data(mech, meta)

            # Drop rows with missing values
            df = df.dropna()

            print("Data preprocessing successful.")
            return df
        except Exception as e:
            logging.error(f"Data preprocessing failed. {str(e)}")
            raise
