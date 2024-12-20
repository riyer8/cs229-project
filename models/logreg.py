import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data.ticker_settings import ALL_TICKERS
from data.vol_loader import generate_ticker_dataset

def process_ticker_classification(start_date, end_date, tickers, output_folder, enable_plotting=True):
    """
    Processes volatility classification using logistic regression for all tickers and saves the reports and metrics.

    Parameters:
        start_date (str): Start date for data retrieval.
        end_date (str): End date for data retrieval.
        tickers (list): List of tickers to process.
        output_folder (str): Parent folder to save the outputs.
        enable_plotting (bool): If True, generates and saves plots.
    """
    visuals_folder = os.path.join(output_folder, "logreg_visuals")
    accuracies_folder = os.path.join(output_folder, "logreg_accuracies")
    datasets_folder = os.path.join(output_folder, "datasets")
    
    os.makedirs(visuals_folder, exist_ok=True)
    os.makedirs(accuracies_folder, exist_ok=True)
    os.makedirs(datasets_folder, exist_ok=True)

    for ticker in tickers:
        print(f"Processing {ticker}...")

        data = generate_ticker_dataset(ticker, start_date, end_date)
        vol_threshold = data['Volatility'].quantile(0.75)
        data['Volatility Class'] = np.where(data['Volatility'] > vol_threshold, 1, 0)  # 1 for high, 0 for low
        data.dropna(inplace=True)

        dataset_path = os.path.join(datasets_folder, f"{ticker}_dataset.csv")
        data.to_csv(dataset_path, index=True)

        features = data[['Daily Return', 'Volatility']]
        target = data['Volatility Class']

        # Training / Testing Set: 70-30
        train_size = int(0.7 * len(features))
        X_train, X_test = features.iloc[:train_size], features.iloc[train_size:]
        y_train, y_test = target.iloc[:train_size], target.iloc[train_size:]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Evaluation Metrics
        print(f"{ticker} - Accuracy: {accuracy}")
        print(f"{ticker} - Classification Report:\n{report}")
        accuracy_path = os.path.join(accuracies_folder, f"{ticker}_classification_report.txt")
        with open(accuracy_path, 'w') as f:
            f.write(f"{ticker} Logistic Regression Info\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Classification Report:\n{report}")

        # Plot Volatility Classification if enabled
        if enable_plotting:
            test_data = data.iloc[train_size:]

            plt.figure(figsize=(12, 6))

            plt.scatter(
                test_data.index, y_test, 
                color='red', label='Actual Class', alpha=0.6
            )
            plt.scatter(
                test_data.index, y_pred, 
                color='blue', label='Predicted Class', alpha=0.6
            )

            plt.title(f'Volatility Classification for {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Volatility Class')
            plt.legend()
            plt.grid()
            plt.tight_layout()

            plot_path = os.path.join(visuals_folder, f"{ticker}_volatility_classification.png")
            plt.savefig(plot_path)
            plt.close()

START_DATE = '2013-01-01'
END_DATE = '2023-12-31'
OUTPUT_FOLDER = "logreg"
process_ticker_classification(START_DATE, END_DATE, ALL_TICKERS, OUTPUT_FOLDER)