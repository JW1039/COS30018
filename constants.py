# Number of prediction steps (days)
K_STEPS = 30 * 6

# Length of the input sequence (e.g., 60 days of past data for model training)
SEQUENCE_LENGTH = 60

# Columns used as features for training
FEATURE_COLUMNS = ['Open', 'Close', 'Interest_Rate']

# Number of time steps for multistep prediction
N_STEPS = 25

# Target column for prediction (must be in FEATURE_COLUMNS)
PREDICTION_COLUMN = 'Close'

# Dictionary to store scalers for feature normalization
SCALERS = {}

# Stock ticker for the company
COMPANY = 'CBA.AX'

# Start date for training data
TRAIN_START = '2015-01-01'

# End date for training data
TRAIN_END = '2017-01-01'
