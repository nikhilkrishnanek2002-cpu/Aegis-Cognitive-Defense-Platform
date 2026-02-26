import sqlite3
try:
    c = sqlite3.connect('results/users.db')
    print("Users:", c.execute('SELECT * FROM users').fetchall())
    c.close()
except Exception as e:
    print("Error:", e)
