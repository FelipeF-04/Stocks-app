import sqlite3

conn = sqlite3.connect("stocks_app/finance.db")
'''conn.executescript("""
CREATE TABLE IF NOT EXISTS users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT    NOT NULL UNIQUE,
    hash     TEXT    NOT NULL,
    cash     NUMERIC NOT NULL DEFAULT 10000.00
);


CREATE TABLE IF NOT EXISTS stocks (
    id     INTEGER NOT NULL,
    symbol TEXT    NOT NULL,
    shares INTEGER NOT NULL,
    total  NUMERIC NOT NULL,
    date   TEXT    NOT NULL,
    time   TEXT    NOT NULL,
    FOREIGN KEY (id) REFERENCES users(id)
);
""")'''

c = conn.cursor()
c.execute("DELETE from users where username = 'feli'")
conn.commit()
conn.close()
