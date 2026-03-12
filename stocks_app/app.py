import sqlite3
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
import requests

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta
import yfinance as yf
import matplotlib.dates as mdates

from prediction_service import predict_stock


def lookup(symbol):
    symbol = symbol.upper().strip()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    
    # Set headers to avoid potential blocking
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check if we got valid data
        if 'chart' not in data or not data['chart']['result']:
            print(f"No data found for symbol: {symbol}")
            return None
            
        result = data['chart']['result'][0]
        meta = result['meta']
        
        return {
            "name": meta.get('shortName', symbol),
            "price": meta.get('regularMarketPrice', 0),
            "symbol": symbol
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except (KeyError, IndexError) as e:
        print(f"Data parsing error: {e}")
    except ValueError as e:
        print(f"JSON decoding error: {e}")
    
    return None


from functools import wraps
from flask import g, request, redirect, url_for

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def usd(value):
    """Format value as USD."""
    return f"${value:,.2f}"

# Configure application
app = Flask(__name__)

# Custom filter

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)



from flask import g

@app.before_request
def open_db():
    g.db = sqlite3.connect("stocks_app/finance.db", isolation_level=None)
    g.db.row_factory = sqlite3.Row

@app.teardown_request
def close_db(exc):
    db = getattr(g, "db", None)
    if db is not None:
        db.close()



@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
@login_required
def index():
    #lookup() returns a dict with keys: "name", "price", "symbol"
    #db.execute select returns a list of dictionaries
    """Show portfolio of stocks"""
    #1st: all_stocks = db.execute("SELECT symbol, shares, total FROM stocks WHERE id = ? ORDER BY shares DESC", session["user_id"])
    all_stocks = g.db.execute("SELECT symbol, SUM(shares) as shares, SUM(total) as total FROM stocks WHERE id = ? GROUP BY symbol ORDER BY total DESC", (session["user_id"],)).fetchall()
    #sqlite> SELECT symbol, SUM(shares) as [shares], SUM(total) as [total] FROM stocks WHERE id = 3 GROUP BY symbol ORDER BY total DESC;
    #https://pythontutor.com/render.html#code=all_stocks%20%3D%20%5B%7B%22price%22%3A%20100,%20%22shares%22%3A%202%7D,%7B%22price%22%3A%20150,%20%22shares%22%3A%203%7D%5D%0A%0Afor%20i%20in%20all_stocks%3A%0A%20%20%20%20i%5B%22total%22%5D%20%3D%20i%5B%22price%22%5D%20*%20i%5B%22shares%22%5D&cumulative=false&curInstr=5&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false
    total = 0
    all_stocks = [dict(i) for i in all_stocks if i["shares"] != 0]
    for i in all_stocks:
        current_cost = lookup(i["symbol"])
        current_cost = current_cost["price"]
        i["mycost"] = usd(i["total"]/i["shares"])

        i["price"] = usd(current_cost)
        i["total"] = int(i["shares"]) * current_cost
        total += i["total"]
        i["total"] = usd(i["total"])

    cash = g.db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchall()
    cash = cash[0]["cash"]
    total += cash
    return render_template("layout.html", data=all_stocks, cash=usd(cash),total=usd(total))


@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    #new table: | person_id | stock_symbol | shares | total | date |
    """Buy shares of stock"""
    if request.method == "POST":
        results = lookup(request.form.get("symbol"))
        if results is None:
            phrase = {"sentence": "Stock not found"}
            return render_template("apology.html", phrase=phrase)
        symbol = results["symbol"]
        price = results["price"]
        shares = request.form.get("shares")
        if shares.isdecimal():
            if int(shares) < 1:
                phrase = {"sentence": "Invalid number of shares to buy"}
                return render_template("apology.html", phrase=phrase)
        else:
            phrase = {"sentence": "Invalid input"}
            return render_template("apology.html", phrase=phrase)
        total = price * int(shares) #this "shares" are not from the database which is indeed an integer, this one is a text one, that is why I use int()

        current_cash = g.db.execute("SELECT * from users WHERE id = ?", (session["user_id"],)).fetchall()
        cash = float(current_cash[0]["cash"])-total
        if cash < 0:
            phrase = {"sentence": "Not enough money to buy"}
            return render_template("apology.html", phrase=phrase)
        g.db.execute("INSERT INTO stocks (id, symbol, shares, total, date, time) VALUES (?, ?, ?, ?, date('now'), time('now'))",(session["user_id"], symbol, shares, total))
        g.db.execute("UPDATE users SET cash = ? WHERE id = ?", (cash, session["user_id"]))
        flash("Bought!")
        return redirect("/")
    else:
        return render_template("buy.html")

@app.route("/buyfast", methods=["POST"])
@login_required
def buyfast():
    if request.method == "POST":
        results = lookup(request.form.get("symbol"))
        if results is None:
            phrase = {"sentence": "Stock not found"}
            return render_template("apology.html", phrase=phrase)
        symbol = results["symbol"]
        price = results["price"]

        current_cash = g.db.execute("SELECT * from users WHERE id = ?", (session["user_id"],)).fetchall()
        cash = current_cash[0]["cash"]-price
        if cash < 0:
            phrase = {"sentence": "Not enough money to buy"}
            return render_template("apology.html", phrase=phrase)
        g.db.execute("INSERT INTO stocks (id, symbol, shares, total, date, time) VALUES (?, ?, ?, ?, date('now'), time('now'))",(session["user_id"], symbol, 1, price))
        g.db.execute("UPDATE users SET cash = ? WHERE id = ?", (cash, session["user_id"]))
        flash(f"Done. Bought for {price}!")
        return redirect("/")


@app.route("/history")
@login_required
def history():
    """Show history of transactions"""
    h = g.db.execute("SELECT symbol, shares, total, date, time FROM stocks WHERE id= ?", (session["user_id"],)).fetchall()
    h = [dict(i) for i in h]
    for i in h:
        if i["total"] < 0 and i["shares"] < 0:
            i["action"] = "Sold"
            i["total"] = i["total"] * -1
            i["shares"] = i["shares"] * -1
        else:
            i["action"] = "Bought"
        mycost = i["total"]/i["shares"]
        i["my_cost"] = mycost

    return render_template("history.html", history=h)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Ensure username was submitted
        if not request.form.get("username"):
            phrase = {"sentence": "Introduce a valid username"}
            flash("Introduce a valid username")
            #return render_template("apology.html", phrase=phrase)
            return render_template("login.html")

        # Ensure password was submitted
        elif not request.form.get("password"):
            phrase = {"sentence": "Introduce a valid password"}
            flash("Introduce a valid password")
            #return render_template("apology.html", phrase=phrase)
            return render_template("login.html")

        # Query database for username
        username = request.form.get("username")
        rows = g.db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)).fetchall()

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(
            rows[0]["hash"], request.form.get("password")
        ):
            phrase = {"sentence": "User does not exist. Try again"}
            flash("User not found. Try again")
            #return render_template("apology.html", phrase=phrase)
            return render_template("login.html")

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]
        #global ids
        #ids = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote."""
    if request.method == "POST":
        s = request.form.get("symbol")
        results = lookup(s)
        if results is None:
            phrase = {"sentence": "Stock not found"}
            return render_template("apology.html", phrase=phrase)
        
        # Store the original price before formatting with usd()
        original_price = results["price"]
        results["price"] = usd(results["price"])
        
        chart_image = generate_stock_chart(s)

        return render_template("quoted.html", results=results, graph=chart_image, symbol=s)
            
    else:
        return render_template("quote.html")


# Add this new route
@app.route("/stock/<symbol>")
@login_required
def stock_detail(symbol):
    """Stock detail page with prediction"""
    # Get basic stock info
    results = lookup(symbol)
    if results is None:
        phrase = {"sentence": "Stock not found"}
        return render_template("apology.html", phrase=phrase)
    
    # Get prediction
    prediction = predict_stock(symbol)
    
    # Generate chart
    chart_image = generate_stock_chart(symbol)
    
    return render_template("stock_detail.html", 
                         results=results, 
                         prediction=prediction,
                         graph=chart_image)


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        try:
            username = request.form.get("username")
            password = request.form.get("password")
            confirmation = request.form.get("confirmation")
            if username == "" or password == "" or confirmation == "":
                phrase = {"sentence": "Invalid registration"}
                return render_template("apology.html", phrase=phrase)
            if password == confirmation:
                g.db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", (username, generate_password_hash(password)))
                rows = g.db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchall()
                session["user_id"] = rows[0]["id"]
                return redirect("/")
            else:
                phrase = {"sentence": "Passwords did not match"}
                return render_template("apology.html", phrase=phrase)
        except ValueError:
            phrase = {"sentence": "I do not know"}
            return render_template("apology.html", phrase=phrase)
    else:
        return render_template("register.html")


@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock"""
    if request.method == "POST":
        results = request.form.get("symbol")
        if results is None:
            phrase = {"sentence": "Stock not found"}
            return render_template("apology.html", phrase=phrase)
        results = lookup(results)
        symbol = results["symbol"]
        mystocks = g.db.execute("SELECT symbol, SUM(shares) as [shares] FROM stocks WHERE id = ? AND symbol = ?", (session["user_id"],symbol)).fetchall()

        price = results["price"]
        shares = request.form.get("shares")
        if shares.isdecimal():
            if int(shares) < 1:
                phrase = {"sentence": "Invalid amount of shares to sell"}
                return render_template("apology.html", phrase=phrase)
        else:
            phrase = {"sentence": "Invalid input"}
            return render_template("apology.html", phrase=phrase)
        shares = int(shares)
        for i in mystocks:
            if i["symbol"] == symbol:
                if shares > i["shares"]:
                    phrase = {"sentence": "Not enough stocks to sell"}
                    return render_template("apology.html", phrase=phrase)
        total = price * shares
        cash = g.db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchall()
        cash = cash[0]["cash"] + total
        g.db.execute("INSERT INTO stocks (id, symbol, shares, total, date, time) VALUES (?, ?, ?, ?, date('now'),time('now'))",(session["user_id"], symbol, -shares, -total))
        g.db.execute("UPDATE users SET cash = ? WHERE id = ?", (cash, session["user_id"],))
        flash(f"Sold for {price}")
        return redirect("/")
    else:
        mystocks = g.db.execute("SELECT symbol, SUM(shares) as shares FROM stocks WHERE id = ? GROUP BY symbol ORDER BY total DESC", (session["user_id"],)).fetchall()
        mystocks = [dict(i) for i in mystocks if i["shares"] != 0]
        for i in mystocks:
            current_cost = lookup(i["symbol"])
            i["price"] = usd(current_cost["price"])
        return render_template("sell.html", data=mystocks)

@app.route("/sellfast", methods=["POST"])
@login_required
def sellfast():
    if request.method == "POST":
        results = lookup(request.form.get("symbol"))
        symbol = results["symbol"]
        mystocks = g.db.execute("SELECT symbol, SUM(shares) as [shares] FROM stocks WHERE id = ? AND symbol = ?", (session["user_id"],symbol)).fetchall()
        price = results["price"]
        for i in mystocks:
            if i["symbol"] == symbol:
                if i["shares"] < 1:
                    phrase = {"sentence": "Not enough stocks to sell"}
                    return render_template("apology.html", phrase=phrase)

        cash = g.db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchall()
        cash = cash[0]["cash"] + price
        g.db.execute("INSERT INTO stocks (id, symbol, shares, total, date, time) VALUES (?, ?, ?, ?, date('now'), time('now'))",(session["user_id"], symbol, -1, -price))
        g.db.execute("UPDATE users SET cash = ? WHERE id = ?", (cash, session["user_id"]))
        flash(f"Sold for {price}")
        return redirect("/")


def generate_stock_chart(s):
    """Helper function to generate stock chart (extracted from your quote function)"""
    try:
        stock = yf.Ticker(s)
        stock_data = stock.history(period="max")
        
        # Check if we have enough data
        if len(stock_data) == 0:
            raise ValueError("No historical data available")
            
        stock_data = stock_data.loc["1990-01-01":].copy()

        # Get the most recent date and calculate start date for 3 months
        end_date = stock_data.index[-1]
        start_date = end_date - timedelta(days=90)
        recent_data = stock_data.loc[start_date:end_date]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recent_data.index, recent_data['Close'], linewidth=2, color="#FF0000")

        # Formatting
        ax.set_facecolor("#FFFFFF")
        fig.patch.set_facecolor("#FFFFFF")
        ax.tick_params(axis="x", colors='black', labelsize = 16)
        ax.tick_params(axis="y", colors='black', labelsize = 16)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Use the actual stock symbol in the title
        ax.set_title(f'{s.upper()} - Last 3 Months', color='black', fontsize=16, fontweight='bold')
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

        # Save to buffer and encode
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Encode the image to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)  # Important: close the figure to free memory
        
        return image_base64
        
    except Exception as e:
        # If chart generation fails, still return the quote without the chart
        print(f"Chart generation error: {e}")
        return None

if __name__ == '__main__':  
   app.run() 

