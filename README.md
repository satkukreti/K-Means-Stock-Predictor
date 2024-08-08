# Money-101

**Buy high, sell low.**  
Use this model to predict whether a stock price will increase or decrease.

## Trading Strategy

### Thought Process

I wanted to create a long-based portfolio, and indicators like RSI and ATR can predict whether a stock has high momentum. For v1, I predict that a previous high RSI and low-moderate ATR will mean that the stock will do well for the next month.

### Execution Process

I downloaded S&P 500 stock constituents and calculated technical indicators like RSI, ATR, Garman-Klass Volatility, Bollinger Bands, etc. I also calculated returns based on various lags and Fama-French factors. Using these features, I calculated the rolling betas, which made 18 unique features in my DataFrame.

I then used an unsupervised machine learning model to apply K-means clustering with predefined centroids for RSI values. From the 4 clusters, I chose the one indicating high RSI values and added them to my "portfolio", which had monthly stock recommendations.

Lastly, I calculated my returns over a 10-year period and benchmarked them to SPY.

Look at v1_rsi&atr_strategy for the comparison.

## Future Features

- Derive a sentiment score using social media-based opinions.
- Use the sentiment score with RSI & ATR for new clustering.
