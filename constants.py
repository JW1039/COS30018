# Number of days to predict into the future (multistep)
K_STEPS = 5  # Predict the next 5 days

# Sequence length (number of past days used for prediction)
SEQUENCE_LENGTH = 25

FEATURE_COLUMNS = ['Close','Open','Mid_Price']
N_STEPS = 25
PREDICTION_COLUMN = 'Close'
SCALERS = {}