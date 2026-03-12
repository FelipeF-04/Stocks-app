import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
from datetime import datetime, timedelta
import threading
import time

# List of popular stocks to pre-train models for
POPULAR_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 'V']

# FIXED BACKTESTING FUNCTIONS
def predict_fold(train, test, predictors, probability_threshold=0.6):
    """
    Train model and make predictions with custom probability threshold for a single fold
    """
    # Create a new model instance for this fold
    fold_model = RandomForestClassifier(
        n_estimators=500, 
        min_samples_split=50, 
        random_state=1,
        n_jobs=-1,
        class_weight="balanced"
    )
    
    fold_model.fit(train[predictors], train["Target"])
    
    # Get prediction probabilities
    pred_probs = fold_model.predict_proba(test[predictors])[:, 1]
    
    # Apply custom threshold
    preds = (pred_probs >= probability_threshold).astype(int)
    
    # Return both binary predictions and probabilities
    preds_series = pd.Series(preds, index=test.index, name="Predictions")
    probs_series = pd.Series(pred_probs, index=test.index, name="Probabilities")
    
    combined = pd.concat([test["Target"], preds_series, probs_series], axis=1)
    return combined, fold_model

def backtest(data, predictors, start=2500, step=250, probability_threshold=0.6):
    """
    Walk-forward backtesting with custom probability threshold
    """
    all_predictions = []
    fold_models = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        # Create features separately for train and test to avoid leakage
        train_features, train_predictors = create_features(train)
        test_features, test_predictors = create_features(test)
        
        # Ensure we have the same predictors in both sets
        common_predictors = list(set(train_predictors) & set(test_predictors))
        
        if len(common_predictors) > 0:
            predictions, fold_model = predict_fold(train_features, test_features, 
                                                 common_predictors, probability_threshold)
            all_predictions.append(predictions)
            fold_models.append(fold_model)
    
    if all_predictions:
        return pd.concat(all_predictions), fold_models
    else:
        return pd.DataFrame(), []

def evaluate_backtest(predictions, fold_models, predictors):
    """
    Evaluate backtest results and print metrics
    """
    if predictions.empty:
        print("No predictions available for evaluation")
        return
    
    # Calculate evaluation metrics
    precision = precision_score(predictions["Target"], predictions["Predictions"], zero_division=0)
    recall = recall_score(predictions["Target"], predictions["Predictions"], zero_division=0)
    f1 = f1_score(predictions["Target"], predictions["Predictions"], zero_division=0)
    auc = roc_auc_score(predictions["Target"], predictions["Probabilities"])
    
    # Calculate confusion matrix
    cm = confusion_matrix(predictions["Target"], predictions["Predictions"])
    tn, fp, fn, tp = cm.ravel()
    
    print("=" * 50)
    print("BACKTEST PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC-ROC: {auc:.3f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.3f}")
    print(f"Baseline (Always Predict Up): {predictions['Target'].mean():.3f}")
    
    # Average feature importance across all folds
    avg_feature_importance = np.mean([model.feature_importances_ for model in fold_models], axis=0)
    
    feature_importance_df = pd.DataFrame({
        'Feature': predictors[:len(avg_feature_importance)],  # Ensure same length
        'Importance': avg_feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "=" * 50)
    print("TOP 10 MOST IMPORTANT FEATURES (AVERAGED ACROSS FOLDS)")
    print("=" * 50)
    print(feature_importance_df.head(10))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'feature_importance': feature_importance_df
    }




def get_model(symbol="SPY"):
    """Get trained model for a specific stock, using general model as fallback"""
    model_path = f"models/{symbol}_model.pkl"
    general_model_path = "models/SPY_model.pkl"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if we have a specific model for this stock
    if os.path.exists(model_path):
        print(f"Loading existing model for {symbol}")
        model, predictors = joblib.load(model_path)
        return model, predictors
    # Check if we have a general model
    elif os.path.exists(general_model_path):
        print(f"Using general model for {symbol}")
        model, predictors = joblib.load(general_model_path)
        return model, predictors
    else:
        # Train a general model as fallback
        print("Training general model (SPY) for first-time use")
        model, predictors = train_new_model("SPY")
        joblib.dump((model, predictors), general_model_path)
        return model, predictors

def train_new_model(symbol="SPY",run_backtest=False):
    """Train a model on a specific stock's historical data"""
    print(f"Training new model for {symbol}")
    
    # Download data for the specific symbol
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period="max")
    
    # Remove unnecessary columns
    for col in ["Dividends", "Stock Splits"]:
        if col in stock_data.columns:
            del stock_data[col]
    
    # Create target variable
    stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)
    stock_data = stock_data.loc["1990-01-01":].copy()
    
    # Create features
    stock_data, predictors = create_features(stock_data)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=500, 
        min_samples_split=50, 
        random_state=1,
        n_jobs=-1,
        class_weight="balanced"
    )
    
    model.fit(stock_data[predictors], stock_data["Target"])

    # Run backtest if requested
    if run_backtest and len(stock_data) > 3000:  # Ensure we have enough data
        print(f"Running backtest for {symbol}...")
        predictions, fold_models = backtest(stock_data, predictors)
        if not predictions.empty:
            evaluate_backtest(predictions, fold_models, predictors)
    
    return model, predictors

'''def create_features(data):
    """Create technical indicators and features for prediction"""
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
    
    # Additional features. # Ensure all features use only past information
    data["Return1"] = data["Close"].pct_change().shift(1)
    data["Volume_Ratio_5"] = data["Volume"] / data["Volume"].rolling(5).mean().shift(1)
    data["Volume_Force"] = data["Volume"] * data["Return1"].abs()
    data["Momentum_5"] = data["Close"].pct_change(5).shift(1)
    data["Momentum_30"] = data["Close"].pct_change(30).shift(1)
    data["Volatility_5"] = data["Return1"].rolling(5).std().shift(1)
    data["Volatility_30"] = data["Return1"].rolling(30).std().shift(1)
    
    # RSI (Relative Strength Index)
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.shift(1)  # Shift here to avoid leakage
    
    data["RSI_14"] = calculate_rsi(data["Close"])
    
    # Create predictor list
    new_predictors = []
    for horizon in horizons:
        new_predictors.append(f"Close_Ratio_{horizon}")
        new_predictors.append(f"Trend_{horizon}")
    
    new_predictors.extend(["Return1", "Volume_Ratio_5", "Volume_Force", "Momentum_5", 
                          "Momentum_30", "Volatility_5", "Volatility_30", "RSI_14"])
    
    # Drop rows with missing values
    data = data.dropna()
    
    return data, new_predictors'''

def create_features(data):
    """Create technical indicators and features for prediction without data leakage"""
    data = data.copy()
    
    # Basic price features - ensure we only use past data
    horizons = [2, 5, 60, 250, 1000]
    
    for horizon in horizons:
        # Close ratio feature - use only past data
        rolling_avg = data["Close"].rolling(horizon).mean().shift(1)
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"].shift(1) / rolling_avg  # Use yesterday's close

        # Trend feature - use only past price movements (not Target)
        trend_column = f"Trend_{horizon}"
        # Calculate daily returns and use them to determine "up" days
        daily_returns = data["Close"].pct_change().shift(1)  # Yesterday's return
        up_days = (daily_returns > 0).astype(int)
        data[trend_column] = up_days.rolling(horizon).sum().shift(1)  # Count of up days
    
    # Additional features - ensure all features use only past information
    data["Return1"] = data["Close"].pct_change().shift(1)  # Yesterday's return
    data["Volume_Ratio_5"] = data["Volume"].shift(1) / data["Volume"].rolling(5).mean().shift(1)  # Use yesterday's volume
    data["Volume_Force"] = data["Volume"].shift(1) * data["Return1"].abs()  # Use yesterday's volume
    data["Momentum_5"] = data["Close"].pct_change(5).shift(1)  # Momentum up to yesterday
    data["Momentum_30"] = data["Close"].pct_change(30).shift(1)  # Momentum up to yesterday
    data["Volatility_5"] = data["Return1"].rolling(5).std().shift(1)  # Volatility up to yesterday
    data["Volatility_30"] = data["Return1"].rolling(30).std().shift(1)  # Volatility up to yesterday
    
    # RSI (Relative Strength Index) - ensure no lookahead
    def calculate_rsi(series, window=14):
        # Use only past data up to the previous day
        delta = series.diff().shift(1)  # Use differences up to previous day
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.shift(1)  # Shift again to ensure no current data
    
    data["RSI_14"] = calculate_rsi(data["Close"])
    
    # MACD (Moving Average Convergence Divergence) - using only past data
    exp12 = data["Close"].ewm(span=12, adjust=False).mean().shift(1)  # Up to yesterday
    exp26 = data["Close"].ewm(span=26, adjust=False).mean().shift(1)  # Up to yesterday
    data["MACD"] = exp12 - exp26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean().shift(1)
    data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]
    
    # Bollinger Bands - using only past data
    data["BB_Middle"] = data["Close"].rolling(20).mean().shift(1)
    bb_std = data["Close"].rolling(20).std().shift(1)
    data["BB_Upper"] = data["BB_Middle"] + (bb_std * 2)
    data["BB_Lower"] = data["BB_Middle"] - (bb_std * 2)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"]
    
    # Create predictor list
    new_predictors = []
    for horizon in horizons:
        new_predictors.append(f"Close_Ratio_{horizon}")
        new_predictors.append(f"Trend_{horizon}")
    
    new_predictors.extend(["Return1", "Volume_Ratio_5", "Volume_Force", "Momentum_5", 
                          "Momentum_30", "Volatility_5", "Volatility_30", "RSI_14",
                          "MACD", "MACD_Signal", "MACD_Histogram", "BB_Middle", 
                          "BB_Upper", "BB_Lower", "BB_Width"])
    
    # Drop rows with missing values
    data = data.dropna()
    
    return data, new_predictors


def predict_stock(symbol):
    """Predict whether a stock will go up tomorrow with detailed recommendations"""
    try:
        # Download enough data for all features (1000 days + buffer)
        stock = yf.Ticker(symbol)
        data = stock.history(period="1100d")  # Enough for 1000-day features
        
        if len(data) < 1000:  # Not enough data for all features
            return {
                "symbol": symbol,
                "error": "Not enough historical data for accurate prediction"
            }
        
        # Prepare data for prediction
        data["Tomorrow"] = data["Close"].shift(-1)
        data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
        data, predictors = create_features(data)
        
        if len(data) == 0:  # Feature creation failed
            return {
                "symbol": symbol,
                "error": "Could not generate features for prediction"
            }
        
        # Get the latest data point for prediction
        latest = data.iloc[[-1]].copy()
        
        # Load appropriate model
        model, predictors = get_model(symbol)
        
        # Make prediction
        prob = model.predict_proba(latest[predictors])[0][1]
        
        # Generate detailed recommendation based on probability
        if prob >= 0.6:
            recommendation = "STRONG BUY"
            confidence = "High"
            color_class = "buy-strong"
        elif prob >= 0.55:
            recommendation = "WEAK BUY"
            confidence = "Medium"
            color_class = "buy-weak"
        elif prob >= 0.45:
            recommendation = "HOLD"
            confidence = "Low"
            color_class = "hold"
        else:
            recommendation = "SELL"
            confidence = "High"
            color_class = "sell"
        
        return {
            "symbol": symbol,
            "prediction": 1 if prob >= 0.5 else 0,
            "probability": prob,
            "confidence": confidence,
            "recommendation": recommendation,
            "color_class": color_class,
            "message": f"Our AI predicts a {prob:.2%} probability of {symbol} increasing tomorrow",
            "error": None
        }
    
    except Exception as e:
        print(f"Prediction error for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": f"Prediction error: {str(e)}"
        }

def pre_train_models():
    """Pre-train models for popular stocks in the background"""
    # Wait a bit to avoid slowing down server startup
    time.sleep(10)
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Pre-train general model if it doesn't exist
    if not os.path.exists("models/SPY_model.pkl"):
        try:
            model, predictors = train_new_model("SPY")
            joblib.dump((model, predictors), "models/SPY_model.pkl")
            print("Pre-trained general market model (SPY)")
        except Exception as e:
            print(f"Error pre-training SPY model: {e}")
    
    # Pre-train models for popular stocks
    for symbol in POPULAR_STOCKS:
        model_path = f"models/{symbol}_model.pkl"
        if not os.path.exists(model_path):
            try:
                model, predictors = train_new_model(symbol)
                joblib.dump((model, predictors), model_path)
                print(f"Pre-trained model for {symbol}")
            except Exception as e:
                print(f"Error pre-training {symbol} model: {e}")

# Start pre-training on import
thread = threading.Thread(target=pre_train_models)
thread.daemon = True
thread.start()


#model, predictors = train_new_model("AAPL", run_backtest=True)

#print(predict_stock("nvda"))