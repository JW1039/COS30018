# Number of days to predict into the future (multistep)
K_STEPS = 7  # Predict the next 5 days

# Sequence length (number of past days used for prediction)
SEQUENCE_LENGTH = 25

FEATURE_COLUMNS = ['Close','Open']
N_STEPS = 25
PREDICTION_COLUMN = 'Close'
SCALERS = {}

# DATA_SOURCE = "yahoo"
COMPANY = 'MSFT'

TRAIN_START = '2022-01-01'     # Start date to read
TRAIN_END = '2023-01-01'       # End date to read

