'''from flask import Flask, send_file
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route("/")
def index():
    return '<img src="/plot.png">'

@app.route("/plot.png")
def plot_png():
    fig, ax = plt.subplots()
    ax.plot([0,1,2],[10,20,15])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    fig.show()
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)


"graph to use"
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 1. DATA DOWNLOAD AND CLEANING
sp500 = yf.Ticker("intc")
sp500_data = sp500.history(period="max")

# Filter data to start from 1990
sp500_data = sp500_data.loc["1990-01-01":].copy()

# 6. CREATE STOCK PRICE CHART FOR LAST 3 MONTHS
# Get the most recent date from the data and calculate start date
end_date = sp500_data.index[-1]
start_date = end_date - timedelta(days=90)

# Filter data for the last 3 months
recent_data = sp500_data.loc[start_date:end_date]

# Get current price and format it
current_price = recent_data['Close'].iloc[-1]
formatted_price = f"${current_price:.2f}"

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot closing price
ax.plot(recent_data.index, recent_data['Close'], linewidth=2, color="#FF0000")

# Formatting for stock app-like appearance
ax.set_facecolor("#FFFFFF")  # Black background
fig.patch.set_facecolor("#FFFFFF")  # Black background for figure
ax.tick_params(colors='black')  # White ticks
ax.spines['bottom'].set_color('black')  # White bottom spine
ax.spines['left'].set_color('black')  # White left spine
ax.spines['top'].set_visible(False)  # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine

# Set title and labels with white color
ax.set_title('INTL (Intel) - Last 3 Months', color='white', fontsize=16, fontweight='bold')
ax.set_ylabel('Price (USD)', color='white', fontsize=12)

# Format x-axis to show dates nicely
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

# Add grid
ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)

# Add percentage change from start to current
start_price = recent_data['Close'].iloc[0]
price_change = ((current_price - start_price) / start_price) * 100
change_color = '#34C759' if price_change >= 0 else '#FF3B30'  # Green for positive, red for negative

# Add annotation for price change
ax.text(0.02, 0.95, f'{price_change:+.2f}%', transform=ax.transAxes, 
        color=change_color, fontsize=14, fontweight='bold',
        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

# Add current price annotation
ax.text(0.98, 0.95, formatted_price, transform=ax.transAxes, 
        color='white', fontsize=16, fontweight='bold',
        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'),
        horizontalalignment='right')

# Add a marker at the current price
last_date = recent_data.index[-1]
ax.plot(last_date, current_price, 'o', color=change_color, markersize=8, 
        markeredgecolor='white', markeredgewidth=1.5)

# Rotate x-axis labels for better readability
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.show()'''


from flask import Flask, send_file
import matplotlib.pyplot as plt
import io
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.dates as mdates

app = Flask(__name__)

@app.route("/")
def index():
    return '<img src="/plot.png">'

@app.route("/plot.png")
def plot_png():
    # Download stock data
    stock = yf.Ticker("intc")
    stock_data = stock.history(period="max")
    stock_data = stock_data.loc["1990-01-01":].copy()

    # Get the most recent date and calculate start date for 3 months
    end_date = stock_data.index[-1]
    start_date = end_date - timedelta(days=90)
    recent_data = stock_data.loc[start_date:end_date]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(recent_data.index, recent_data['Close'], linewidth=2, color="#FF0000")

    # Formatting
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('INTL (Intel) - Last 3 Months', color='black', fontsize=16, fontweight='bold')
    ax.set_ylabel('Price (USD)', color='black', fontsize=12)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    # Add grid
    ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)

    # Add price change and current price annotations
    current_price = recent_data['Close'].iloc[-1]
    start_price = recent_data['Close'].iloc[0]
    price_change = ((current_price - start_price) / start_price) * 100
    change_color = '#34C759' if price_change >= 0 else '#FF3B30'

    ax.text(0.02, 0.95, f'{price_change:+.2f}%', transform=ax.transAxes, 
            color=change_color, fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    formatted_price = f"${current_price:.2f}"
    ax.text(0.98, 0.95, formatted_price, transform=ax.transAxes, 
            color='black', fontsize=16, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            horizontalalignment='right')

    # Add a marker at the current price
    last_date = recent_data.index[-1]
    ax.plot(last_date, current_price, 'o', color=change_color, markersize=8, 
            markeredgecolor='black', markeredgewidth=1.5)

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

    # Save to buffer and return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)

