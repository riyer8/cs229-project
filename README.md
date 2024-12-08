# StockVol: Predicting Prices & Volatility with News & Market Data

## Project Category: Financial Machine Learning

Created by: Ramya Iyer, Jenny Wei, and Tatiana Zhang

## Overview

StockVol is a machine learning architecture designed to predict stock market volatility by integrating both quantitative market data and qualitative sentiment analysis derived from financial news and social media. The architecture combines traditional econometric models, like GARCH, with advanced neural network-based models, such as Long Short-Term Memory (LSTM) networks. By fusing financial metrics and sentiment data, StockVol outperforms baseline models in accuracy when forecasting volatility, offering a comprehensive tool for investors and market analysts. This approach bridges the gap between structured financial analysis and unstructured sentiment data, addressing the challenges of volatility prediction in dynamic markets.

## Key Components

- **Quantitative Features:** Includes daily stock data such as open, high, low, close prices, trading volume, and derived metrics like daily variation, moving averages, and volatility indicators.
- **Qualitative Sentiment Analysis:** Sentiment scores derived from financial news using the FinBERT model to measure daily market sentiment on a scale from -1 (negative) to 1 (positive).
- **Baseline Models:** Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models, linear regression, logistic regression, and polynomial regression for fundamental volatility prediction.
- **Advanced Models:** LSTM and GRU networks designed to capture long-term dependencies in sequential data, integrating both quantitative and qualitative features for enhanced predictions.

## Methodology

1. **Dataset and Features:**
    - **Source:** Stock data for companies like Apple (AAPL) and indices such as the Dow Jones (DOW) from Yahoo Finance via the yfinance API (January 1, 2006 to December 31, 2020).
    - **Train / Validation / Test Split:** 70/20/10 for robust time-series modeling.
    - **Quantitative Features:**
        - Metrics: Daily variation, returns, moving averages (7-day SMA, EMA), volatility measures (standard deviation, ATR), and technical indicators (MACD, RSI).
        - Feature Engineering: Applied correlation analysis, mutual information, and PCA to refine features and reduce dimensionality.
    - **Sentiment Data:**
        - Financial news from Reuters, Bloomberg (2006-2013), and Kaggle datasets (2013-2020).
        - Preprocessing: Text normalization, tokenization, and sentiment scoring with FinBERT.

2. **Baseline Models:**
    - **GARCH (1,1):** Predicts volatility using past squared residuals and conditional variances.
    $\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \epsilon_{t-1}^2$
    - **Linear, Logistic, and Polynomial Regression:** Simple models to capture relationships in stock behavior and volatility direction.

3. **Advanced Models:**
    - **LSTM Network:** Processes sequential data with hidden states and cell states to capture long-term dependencies.
        - Architecture: Single-layer LSTM with 32 hidden units followed by a dense layer.
    - **GRU Network:** Simplifies LSTM by combining gates for reduced complexity.

4. **Sentiment Analysis:**
    - **Model:** FinBERT for sentiment scoring.
    - **Preprocessing:** Lowercasing, removal of URLs, HTML tags, and non-alphanumeric characters.
    - **Sentiment Aggregation:** Averaging daily sentiment scores to align with financial data.

## Results

### Performance

- **LSTM and GRU models** outperformed baseline models (linear regression and GARCH), especially when sentiment features were included.
- **GRU with sentiment** achieved the **best performance**, with the lowest Mean Squared Error (MSE) and highest \(R^2\) and correlation scores.

### Insights

1. **Feature Effectiveness**:  
   The combination of **Daily Return**, **High-Close**, and **Low-Open** features proved most effective. Adding sentiment data improved accuracy further.

2. **Sentiment Integration**:  
   Including sentiment analysis enhanced performance, reducing MSE and increasing the explained variance for both LSTM and GRU models.

3. **Model Comparison**:  
   - **GRU** slightly outperformed **LSTM**, particularly with sentiment data.  
   - Baseline models (linear regression and GARCH) were **less accurate** but provided interpretable results.

---

### Detailed Results Table

| **Model**                        | **MSE**   | **MAE**   | **\(R^2\)**    | **Correlation Coefficient** | **Explained Variance** |
|----------------------------------|-----------|-----------|----------------|-----------------------------|------------------------|
| **LSTM (Without Sentiment)**     | 14.3929   | 2.6345    | 0.6645         | 0.8337                      | 0.6824                 |
| **LSTM (With Sentiment)**        | 13.6513   | 2.3224    | 0.6953         | 0.8691                      | 0.6879                 |
| **GRU (Without Sentiment)**      | 13.7926   | 2.8162    | 0.6785         | 0.8401                      | 0.7009                 |
| **GRU (With Sentiment)**         | 12.1735   | 2.7215    | 0.7099         | 0.8472                      | 0.7104                 |
| **Linear Regression (DJI)**      | N/A       | 25.3224   | N/A            | -0.0192                     | N/A                    |

## Future Work

- **Extended Finetuning**:  
  We plan to extend the training period and incorporate larger datasets for both the classifier and specialized models. This will allow us to fine-tune the models further, improving their generalization and accuracy in predicting stock market volatility.

- **Granular Categorization**:  
  We aim to implement more detailed categorization of sentiment and stock volatility-related features to reduce ambiguity and enhance robustness in our predictions. By refining the feature sets and model training, we can better capture market complexities.

- **Sentiment Expansion**:  
  Future work will involve expanding sentiment analysis to incorporate more sources, such as social media, analyst reports, and financial podcasts. This could provide a richer, multi-dimensional understanding of market sentiment and improve predictions.

- **Hybrid Models**:  
  We plan to experiment with hybrid models that combine LSTM/GRU architectures with autoregressive techniques, reinforcement learning, or graph-based methods, enabling a more sophisticated understanding of the market's temporal dependencies and complex structures.

- **Larger Models**:  
  Given sufficient computational resources, we intend to train larger, domain-specific models focused on financial texts. This would help capture even subtler linguistic cues influencing market volatility.

## Conclusion

In this study, we explored the use of advanced deep learning architectures—LSTMs and GRUs—for predicting stock market volatility. By incorporating sentiment analysis from financial news articles, our models surpassed traditional methods like GARCH and linear regressions. Notably, GRU models outperformed LSTM models in terms of prediction accuracy and error metrics, thanks to their simplified gating mechanisms that reduce overfitting risks and improve training efficiency.

Our results emphasize the importance of qualitative data, such as sentiment, in enhancing predictive stability and providing additional insights into market behavior. This work opens the door to more refined predictive models, which can leverage both quantitative and qualitative data to better understand and anticipate market volatility.

## Acknowledgements

- **Yahoo Finance** and **Kaggle** for providing financial datasets and news sources
- **Reuters** and **Bloomberg** for access to historical financial news data

## Made in Stanford CS 229 (Machine Learning)
