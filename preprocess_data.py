import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging

def preprocess_data(mech_csv, meta_csv):
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    try:
        
        mech = pd.read_csv(mech_csv)
        meta = pd.read_csv(meta_csv)

        # Height, mass and age are fetched from metadata csv
        df = mech.join(meta[['session_height_m', 'session_mass_kg', 'age_yrs']])

        new_columns = ['session_height_m', 'session_mass_kg', 'age_yrs']

        for col in new_columns:
            df.insert(2, col, df.pop(col))

        
        column_to_move = df['pitch_speed_mph']
        df.drop(columns=['pitch_speed_mph', 'session_pitch', 'session', 'pitch_type', 'p_throws'], inplace=True)
        df.insert(0, 'pitch_speed_mph', column_to_move) # Target variable moved to first position

        # Drop 8 rows with missing values
        df = df.dropna()

        logging.info("Data preprocessing successful.")

        return df

    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        
        raise
