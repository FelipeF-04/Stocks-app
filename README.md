# Stocks App

A web application for managing a stock portfolio with real-time data and price predictions.

## What it does

- Create an account and start with virtual cash ($10,000)
- Look up stock prices in real-time
- Buy and sell stocks to build your portfolio
- View your transaction history
- Get AI-powered price predictions for stocks

## Tech Stack

**Backend:**
- Flask - web framework
- SQLite - database for users, stocks, and transactions
- Flask-Session - session management for user authentication

**Data & APIs:**
- Yahoo Finance API (yfinance) - real-time stock data
- Pandas & NumPy - data manipulation and analysis

**Machine Learning:**
- Scikit-learn - RandomForestClassifier for price predictions
- Joblib - model serialization

**Frontend:**
- HTML/CSS templates (Jinja2)
- Matplotlib - chart generation for visualization
- Bootstrap - responsive styling

## Project Structure

```
stocks_app/
├── app.py                  # Main Flask application
├── prediction_service.py   # ML model for stock predictions
├── sql.py                  # Database utilities
├── static/
│   └── styles.css         # Custom styling
├── templates/             # HTML pages
│   ├── layout.html        # Base template
│   ├── index.html         # Dashboard
│   ├── login.html
│   ├── register.html
│   ├── quote.html         # Stock lookup
│   ├── buy.html           # Purchase interface
│   ├── sell.html          # Sell interface
│   ├── history.html       # Transaction history
│   └── stock_detail.html  # Stock details with predictions
└── finance.db             # SQLite database
```

## Getting Started

1. **Install dependencies:**
   ```
   pip install flask flask-session yfinance pandas numpy scikit-learn matplotlib requests werkzeug
   ```

2. **Run the app:**
   ```
   python stocks_app/app.py
   ```

3. **Access it:**
   Open your browser and go to `http://localhost:5000`

## Features

- **Authentication** - Secure user registration and login
- **Stock Lookup** - Real-time stock prices via Yahoo Finance
- **Portfolio Management** - Buy/sell stocks and track holdings
- **Transaction History** - View all trading activity
- **Price Predictions** - ML-based forecasting for stock movements
- **Charts & Visualization** - Visual representation of stock data

## Notes

- Starting virtual cash: $10,000
- Uses a RandomForest model trained on historical stock data
- Session management keeps users logged in securely
