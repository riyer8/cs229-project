import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data.ticker_settings import ALL_TICKERS
from data.vol_loader import generate_ticker_dataset

def process_ticker_volatility(start_date, end_date, tickers, output_folder, enable_plotting=True):
    """
    Processes volatility prediction using linear regression for all tickers and saves the plots and metrics.

    Parameters:
        start_date (str): Start date for data retrieval.
        end_date (str): End date for data retrieval.
        tickers (list): List of tickers to process.
        output_folder (str): Parent folder to save the outputs.
        enable_plotting (bool): If True, generates and saves plots.
    """
    visuals_folder = os.path.join(output_folder, "linreg_visuals")
    accuracies_folder = os.path.join(output_folder, "linreg_accuracies")
    datasets_folder = os.path.join(output_folder, "datasets")
    
    os.makedirs(visuals_folder, exist_ok=True)
    os.makedirs(accuracies_folder, exist_ok=True)
    os.makedirs(datasets_folder, exist_ok=True)

    for ticker in tickers:
        print(f"Processing {ticker}...")

        data = generate_ticker_dataset(ticker, start_date, end_date)
        data.dropna(inplace=True)
        dataset_path = os.path.join(datasets_folder, f"{ticker}_dataset.csv")
        data.to_csv(dataset_path, index=True)

        features = data[['Daily Return']]
        target = data['Volatility']

        # Training / Testing Set: 70-30
        train_size = int(0.7 * len(features))
        X_train, X_test = features.iloc[:train_size], features.iloc[train_size:]
        y_train, y_test = target.iloc[:train_size], target.iloc[train_size:]

        # Training and Predicting
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Evaluation Metrics
        print(f"{ticker} - Mean Squared Error (MSE): {mse}")
        print(f"{ticker} - R² (Coefficient of Determination): {r2}")
        accuracy_path = os.path.join(accuracies_folder, f"{ticker}_accuracy.txt")
        with open(accuracy_path, 'w') as f:
            f.write(f"{ticker} Linear Regression Info\n")
            f.write(f"R² Score: {r2}\n")
            f.write(f"MSE: {mse}\n")

        # Plot Volatility if enabled
        if enable_plotting:
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

            plt.figure(figsize=(12, 6))

            plt.scatter(
                train_data.index, train_data['Volatility'], 
                color='green', label='Training Data', alpha=0.6
            )
            plt.scatter(
                test_data.index, test_data['Volatility'], 
                color='red', label='Testing Data', alpha=0.6
            )
            plt.plot(
                test_data.index, y_pred, 
                color='blue', label='Predicted Volatility', linewidth=2
            )

            plt.title(f'Volatility Prediction for {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid()
            plt.tight_layout()

            plot_path = os.path.join(visuals_folder, f"{ticker}_volatility_prediction.png")
            plt.savefig(plot_path)
            plt.close()

START_DATE = '2013-01-01'
END_DATE = '2023-12-31'
OUTPUT_FOLDER = "linreg"
process_ticker_volatility(START_DATE, END_DATE, ALL_TICKERS, OUTPUT_FOLDER)