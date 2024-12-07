import csv

from models.lstm_predictor import LSTMVolatilityPredictor
from data import vol_loader

TICKERS = ["^DJI", "AAPL"]
FEATURES = [
    ['Daily Return'],
    ['Daily Return', 'High-Close', 'Low-Open'],  # picking based on low correlation
    ['Daily Return', 'High-Close', 'Low-Open', 'Close Change'],  # picking based on low correlation
    ['Daily Return', 'High-Close', 'Low-Open', 'ADX', 'RSI', 'Daily Variation'],  # picking based on low correlation
    ['Daily Return', 'High-Close', 'Low-Open', 'ADX', 'RSI', 'Daily Variation', 'Stochastic Oscillator'],  # picking based on low mi
    ['Daily Return', 'Volume', 'Daily Variation', 'ADX', '7-Day STD', 'Stochastic Oscillator', 'RSI', 'Low-Open', 'Close % Change', 'High-Close', 'Close Change'],  # picking based on low mi
    ['MACD', 'Close Change', 'RSI', 'Stochastic Oscillator', 'ADX'], # PCA
    ['Daily Return', 'High-Close', 'Low-Open', 'ADX', 'RSI', 'Daily Variation'], # PCA
    ['+DI', 'ATR', 'MACD', '-DI', '14-Day EMA'], # MI + PCA
    ['+DI', 'ATR', 'MACD', '-DI', '14-Day EMA', 'Daily Return'], # MI + PCA
    ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume',
       'Daily Return', 'Volatility', 'Daily Variation', '7-Day SMA',
       '7-Day STD', 'SMA + 2STD', 'SMA - 2STD', 'High-Close', 'Low-Open',
       'Cumulative Return', '14-Day EMA', 'Close % Change', 'Close Change',
       'RSI', 'MACD', 'Stochastic Oscillator', 'ATR', '+DI', '-DI', 'ADX'] # ALL
]

ticker = "^DJI"
data = vol_loader.load_ticker_data(ticker)

def feature_tuning():
    filename = "data/lstm_features.csv"
    best_feature_combo = None
    best_metric_value = float('inf')  # Initialize for minimum metric (e.g., MSE)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Feature Set", "MSE", "MAE", "R-squared", "Correlation Coefficient", "Explained Variance"])

        for feature_set in FEATURES:
            print(f"\nStarting feature set {feature_set} on ticker {ticker}")
            predictor = LSTMVolatilityPredictor(sequence_length=30)
            train_loader, val_loader, test_loader, dates_val_seq, y_val_seq, dates_test_seq, y_test_seq = predictor.prepare_data(
                data=data,
                features=feature_set,
                target='Volatility'
            )
            predictor.build_model(input_dim=len(feature_set))
            predictor.train(train_loader)

            dates, y_true, y_pred, metrics = predictor.evaluate(val_loader, y_val_seq, dates_val_seq)
            writer.writerow([feature_set, metrics["Mean Squared Error"], metrics["Mean Absolute Error"],
                                     metrics["R-squared"], metrics["Correlation Coefficient"], metrics["Explained Variance"]])
            if metrics["Mean Absolute Error"] < best_metric_value:
                best_metric_value = metrics["Mean Absolute Error"]
                best_feature_combo = feature_set
    print(f"Best Feature Combo: {best_feature_combo} with MAE {best_metric_value}")


# feature_tuning()
# Best Feature Combo: ['Daily Return', 'High-Close', 'Low-Open'] with MAE 2.5226359367370605

best_feature = ['Daily Return', 'High-Close', 'Low-Open']
def hidden_size_tuning():
    filename = "data/lstm_hidden_tuning.csv"
    best_hidden_dim = None
    best_metric_value = float('inf')  # Initialize for minimum metric (e.g., MAE)

    hidden_dims = [16, 32, 50, 64, 128]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Hidden Dim", "MSE", "MAE", "R-squared", "Correlation Coefficient", "Explained Variance"])

        for dim in hidden_dims:
            print(f"\nTesting hidden_dim={dim}")
            predictor = LSTMVolatilityPredictor(sequence_length=30, hidden_dim=dim)
            train_loader, val_loader, test_loader, dates_val_seq, y_val_seq, dates_test_seq, y_test_seq = predictor.prepare_data(
                data=data,
                features=best_feature,
                target='Volatility'
            )
            predictor.build_model(input_dim=len(best_feature))
            predictor.train(train_loader)

            dates, y_true, y_pred, metrics = predictor.evaluate(val_loader, y_val_seq, dates_val_seq)

            # Save metrics to file
            writer.writerow([dim, metrics["Mean Squared Error"], metrics["Mean Absolute Error"],
                             metrics["R-squared"], metrics["Correlation Coefficient"], metrics["Explained Variance"]])

            # Track the best hidden_dim
            if metrics["Mean Absolute Error"] < best_metric_value:
                best_metric_value = metrics["Mean Absolute Error"]
                best_hidden_dim = dim

    print(f"Best Hidden Dim: {best_hidden_dim} with MAE {best_metric_value}")

# hidden_size_tuning()
# Best Hidden Dim: 32 with MAE 2.413968563079834
hidden_dim = 32

def run_test():
    predictor = LSTMVolatilityPredictor(sequence_length=30, hidden_dim=hidden_dim)
    train_loader, val_loader, test_loader, dates_val_seq, y_val_seq, dates_test_seq, y_test_seq = predictor.prepare_data(
        data=data,
        features=best_feature,
        target='Volatility'
    )
    predictor.build_model(input_dim=len(best_feature))
    predictor.train(train_loader)
    print("\nValidation Metrics:\n")
    dates_val, y_true_val, y_pred_val, metrics_val = predictor.evaluate(val_loader, y_val_seq, dates_val_seq)

    print("\nTest Metrics:\n")
    dates_test, y_true_test, y_pred_test, metrics_test = predictor.evaluate(val_loader, y_val_seq, dates_val_seq)
    predictor.plot_loss()
    predictor.plot_predictions(dates_test, y_true_test, y_pred_test)


run_test()