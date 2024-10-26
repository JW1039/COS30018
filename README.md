# COS30018 Stock Prediction System
An artificial intelligence stock prediction system with the ability to predict the future stock price of a company given the historical stock data, using Python modifying a pre-created code base.

### Getting Started
To get started with the stock prediction system, you will need Python 3.7+ and the required libraries specified in the `requirements.txt` file.

### Installation
1. Clone the repository or download the source code.
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Program
Once the virtual environment is set up and dependencies are installed, you can run the program with:
```bash
python stock_prediction.py
```

### Customizing Constants
For additional customization, you can edit `constants.py` to adjust various settings:
- **K_STEPS**: Number of prediction steps (days)
- **SEQUENCE_LENGTH**: Sequence length for training the models (days)
- **FEATURE_COLUMNS**: Data columns used as features
- **PREDICTION_COLUMN**: Column to predict (must be part of FEATURE_COLUMNS)
- **COMPANY**: Stock ticker for the company to analyze
- **TRAIN_START** and **TRAIN_END**: Date range for training data
