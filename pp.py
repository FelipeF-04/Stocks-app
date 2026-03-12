import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. DATA DOWNLOAD AND CLEANING
sp500 = yf.Ticker("intc")
sp500_data = sp500.history(period="max")

# Remove unnecessary columns
del sp500_data["Dividends"]
del sp500_data["Stock Splits"]

# Create target variable - whether price will go up tomorrow
sp500_data["Tomorrow"] = sp500_data["Close"].shift(-1)
sp500_data["Target"] = (sp500_data["Tomorrow"] > sp500_data["Close"]).astype(int)

# Filter data to start from 1990
sp500_data = sp500_data.loc["1990-01-01":].copy()

# 2. FEATURE ENGINEERING FUNCTION
def create_features(data):
    """
    Creates technical indicators and features for prediction
    Avoids data leakage by ensuring all features use only past information
    """
    data = data.copy()
    
    # Basic price features
    horizons = [2, 5, 60, 250, 1000]
    
    for horizon in horizons:
        # Close ratio feature
        rolling_avg = data["Close"].rolling(horizon).mean().shift(1)
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_avg

        # Trend feature (count of up days)
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data["Target"].rolling(horizon).sum().shift(1)
    
    # Additional features that improve predictive power
    # Daily return
    data["Return1"] = data["Close"].pct_change().shift(1)
    
    # Volume indicators
    data["Volume_Ratio_5"] = data["Volume"] / data["Volume"].rolling(5).mean().shift(1)
    data["Volume_Force"] = data["Volume"] * data["Return1"].abs()
    
    # Momentum indicators
    data["Momentum_5"] = data["Close"].pct_change(5).shift(1)
    data["Momentum_30"] = data["Close"].pct_change(30).shift(1)
    
    # Volatility indicators
    data["Volatility_5"] = data["Return1"].rolling(5).std().shift(1)
    data["Volatility_30"] = data["Return1"].rolling(30).std().shift(1)
    
    # RSI (Relative Strength Index)
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    data["RSI_14"] = calculate_rsi(data["Close"]).shift(1)
    
    # Create predictor list
    new_predictors = []
    for horizon in horizons:
        new_predictors.append(f"Close_Ratio_{horizon}")
        new_predictors.append(f"Trend_{horizon}")
    
    new_predictors.extend(["Return1", "Volume_Ratio_5", "Volume_Force", "Momentum_5", 
                          "Momentum_30", "Volatility_5", "Volatility_30", "RSI_14"])
    
    # Drop rows with missing values
    data = data.dropna()
    
    return data, new_predictors

# Apply feature engineering
sp500_data, predictors = create_features(sp500_data)

# 3. MODEL DEFINITION
model = RandomForestClassifier(
    n_estimators=500, 
    min_samples_split=50, 
    random_state=1,
    n_jobs=-1,
    class_weight="balanced"
)

# 4. BACKTESTING FUNCTIONS
def predict(train, test, predictors, model, probability_threshold=0.6):
    """
    Train model and make predictions with custom probability threshold
    """
    model.fit(train[predictors], train["Target"])
    
    # Get prediction probabilities
    pred_probs = model.predict_proba(test[predictors])[:, 1]
    
    # Apply custom threshold
    preds = (pred_probs >= probability_threshold).astype(int)
    
    # Return both binary predictions and probabilities
    preds_series = pd.Series(preds, index=test.index, name="Predictions")
    probs_series = pd.Series(pred_probs, index=test.index, name="Probabilities")
    
    combined = pd.concat([test["Target"], preds_series, probs_series], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250, probability_threshold=0.6):
    """
    Walk-forward backtesting with custom probability threshold
    """
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        predictions = predict(train, test, predictors, model, probability_threshold)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

# 5. RUN BACKTEST AND EVALUATE
predictions = backtest(sp500_data, model, predictors, probability_threshold=0.6)

# Calculate evaluation metrics
precision = precision_score(predictions["Target"], predictions["Predictions"])
recall = recall_score(predictions["Target"], predictions["Predictions"])
f1 = f1_score(predictions["Target"], predictions["Predictions"])
auc = roc_auc_score(predictions["Target"], predictions["Probabilities"])

# Calculate confusion matrix
cm = confusion_matrix(predictions["Target"], predictions["Predictions"])
tn, fp, fn, tp = cm.ravel()

# 6. STRATEGY PERFORMANCE ANALYSIS
# FIXED: Correct way to get the first and last close prices
first_date = predictions.index[0]
last_date = predictions.index[-1]
first_close = sp500_data.loc[first_date, 'Close']
last_close = sp500_data.loc[last_date, 'Close']

# Simulate trading based on predictions
'''initial_capital = 10000
capital = initial_capital
position = 0  # 0 = no position, 1 = long position
trade_log = []
entry_price = 0

for i, row in predictions.iterrows():
    current_close = sp500_data.loc[i, 'Close']
    
    if row['Predictions'] == 1 and position == 0:
        # Buy signal: enter long position
        entry_price = current_close
        position = 1
        trade_log.append({'Date': i, 'Action': 'BUY', 'Price': entry_price, 'Capital': capital})
    elif (row['Predictions'] == 0 or i == predictions.index[-1]) and position == 1:
        # Sell signal: exit position
        exit_price = current_close
        capital *= (exit_price / entry_price)  # Update capital based on return
        position = 0
        trade_log.append({'Date': i, 'Action': 'SELL', 'Price': exit_price, 'Capital': capital})

# Calculate strategy returns
strategy_return = (capital - initial_capital) / initial_capital * 100

# Buy-and-hold benchmark
buy_hold_return = (last_close / first_close - 1) * 100
'''
# 7. PRINT RESULTS
print("=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"AUC-ROC: {auc:.3f}")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.3f}")
'''
print("\n" + "=" * 50)
print("TRADING STRATEGY PERFORMANCE")
print("=" * 50)
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Capital: ${capital:,.2f}")
print(f"Strategy Return: {strategy_return:.2f}%")
print(f"Buy-and-Hold Return: {buy_hold_return:.2f}%")
print(f"Number of Trades: {len([t for t in trade_log if t['Action'] == 'BUY'])}")
'''
print("\n" + "=" * 50)
print("PREDICTION DISTRIBUTION")
print("=" * 50)
print(predictions["Predictions"].value_counts())
print(f"Baseline (Always Predict Up): {predictions['Target'].mean():.3f}")

# 8. FEATURE IMPORTANCE ANALYSIS
feature_importance = pd.DataFrame({
    'Feature': predictors,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "=" * 50)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 50)
print(feature_importance.head(10))

'''# 9. VISUALIZATION (Optional)
if trade_log:
    trade_results = pd.DataFrame(trade_log)
    buy_actions = trade_results[trade_results['Action'] == 'BUY']
    sell_actions = trade_results[trade_results['Action'] == 'SELL']
    
    plt.figure(figsize=(12, 6))
    plt.plot(trade_results['Date'], trade_results['Capital'], label='Strategy Equity')
    
    if not buy_actions.empty:
        plt.scatter(buy_actions['Date'], buy_actions['Capital'], color='green', marker='^', label='Buy', s=100)
    if not sell_actions.empty:
        plt.scatter(sell_actions['Date'], sell_actions['Capital'], color='red', marker='v', label='Sell', s=100)
    
    plt.title('Trading Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No trades were executed during the backtest period.")'''