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
- Bootstrap-style layout and components in templates

## Project Structure

```
.
├── .gitignore
├── README.md
├── models/                 # Pretrained model .pkl files
└── stocks_app/
    ├── app.py              # Main Flask application
    ├── prediction_service.py
    ├── sql.py
    ├── finance.db          # SQLite database used by the app
    ├── static/
    │   └── styles.css
    └── templates/
        ├── layout.html
        ├── login.html
        ├── register.html
        ├── quote.html
        ├── quoted.html
        ├── buy.html
        ├── sell.html
        ├── history.html
        ├── stock_detail.html
        └── apology.html
```

## Getting Started

1. **Open a terminal at the repository root**
   This must be the folder that contains `README.md`.

2. **Create and activate a virtual environment (recommended):**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```
   pip install flask flask-session yfinance pandas numpy scikit-learn matplotlib requests werkzeug joblib
   ```

4. **Run the app:**
   ```
   python stocks_app/app.py
   ```

5. **Access it:**
   Open your browser and go to `http://localhost:5000`

## Database Note (Important for Other Devices)

- This app opens `stocks_app/finance.db` directly.
- The repo currently includes that file, so cloning should run immediately.
- If `finance.db` is removed, the app will fail until you recreate the `users` and `stocks` tables.

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
- Session data is stored by Flask-Session in `flask_session/` and should remain ignored
- Run from the repo root so relative paths like `stocks_app/finance.db` resolve correctly
