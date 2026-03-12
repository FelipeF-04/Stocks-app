'''import yfinance as yf

sp500 = yf.Ticker("NFLX")

sp500 = sp500.history(period="max")
#print(sp500)

#print(sp500.plot.line(y="Close", use_index=True))

del sp500["Dividends"]
del sp500["Stock Splits"]


sp500["Tomorrow"] = sp500["Close"].shift(-1)


sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


sp500 = sp500.loc["1990-01-01":].copy()

#print(sp500)

from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score
import pandas as pd
preds = model.predict(test[predictors])
preds = pd.Series (preds, index=test.index)

#print(precision_score(test["Target"], preds))

combined = pd.concat([test["Target"], preds], axis=1)


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)

#print(predictions["Predictions"].value_counts())

#print(precision_score(predictions["Target"], predictions["Predictions"]))
#print(predictions["Target"].value_counts()/predictions.shape[0])


horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]


    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()
#print(sp500)


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0


    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = backtest(sp500, model, new_predictors)
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))'''

# --- START: Safe feature build + live predictions for 1-day and 7-day horizons ---
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import yfinance as yf

sp500 = yf.Ticker("trib")

sp500 = sp500.history(period="max")


del sp500["Dividends"]
del sp500["Stock Splits"]


sp500["Tomorrow"] = sp500["Close"].shift(-1)


sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


sp500 = sp500.loc["1990-01-01":].copy()

# base predictors (same names you used originally)
predictors = ["Close", "Volume", "Open", "High", "Low"]

# 1) Targets: 1-day and 7-day future direction
sp500["Target"]   = (sp500["Close"].shift(-1) > sp500["Close"]).astype(int)   # tomorrow up?
sp500["Target_7"] = (sp500["Close"].shift(-7) > sp500["Close"]).astype(int)   # 7 days ahead up?

# 2) Build rolling-based features (safe: use only past values -> shift(1) before rolling)
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    # rolling mean of past closes only (shift(1) ensures we don't include current day's close)
    rolling_mean = sp500["Close"].shift(1).rolling(window=horizon).mean()
    ratio_col = f"Close_Ratio_{horizon}"
    sp500[ratio_col] = sp500["Close"] / rolling_mean
    # trend: count of previous "up" days in the last `horizon` days (use Target shifted to avoid using today's target)
    trend_col = f"Trend_{horizon}"
    sp500[trend_col] = sp500["Target"].shift(1).rolling(window=horizon).sum()
    new_predictors += [ratio_col, trend_col]

# 3) Add a few small extras you can reuse (lagged returns, momentum, volatility) safely
sp500["ret_1"] = sp500["Close"].pct_change().shift(1)
sp500["mom_5"] = sp500["Close"].pct_change(5).shift(1)
sp500["vol_10"] = sp500["Close"].pct_change().shift(1).rolling(10).std()

extras = ["ret_1", "mom_5", "vol_10"]
new_predictors += extras

# 4) Final predictors set used for training/prediction
final_predictors = predictors + new_predictors

# 5) Drop rows with NaNs in the features or targets (must happen after targets created)
sp500_safe = sp500.dropna(subset=final_predictors + ["Target", "Target_7"]).copy()

# 6) Train a model for H=1 (tomorrow) and H=7 (one week) using all available past rows
model_1 = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
model_7 = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

model_1.fit(sp500_safe[final_predictors], sp500_safe["Target"])
model_7.fit(sp500_safe[final_predictors], sp500_safe["Target_7"])

# 7) Build the "most recent features" row to produce a live prediction.
#    Use the last available row from the original sp500 (not necessarily sp500_safe),
#    but ensure that row has no NaNs for the features.
last_row = sp500.iloc[[-1]][final_predictors]

if last_row.isnull().any(axis=None):
    print("Last row has NaNs in features — need more history before making a live prediction.")
else:
    # Probability of up for each horizon
    prob_up_1 = model_1.predict_proba(last_row)[:,1][0]
    prob_up_7 = model_7.predict_proba(last_row)[:,1][0]

    # Use same thresholding approach you used before (0.7) or pick 0.5 for balanced decision
    threshold = 0.6
    pred_1 = 1 if prob_up_1 >= threshold else 0
    pred_7 = 1 if prob_up_7 >= threshold else 0

    print(f"Tomorrow (H=1):  P(up) = {prob_up_1:.3f}  -> predicted class (threshold {threshold}) = {pred_1}")
    print(f"In 7 days (H=7): P(up) = {prob_up_7:.3f}  -> predicted class (threshold {threshold}) = {pred_7}")

# --- END snippet ---

