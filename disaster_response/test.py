from logging import debug
from flask import Flask
from werkzeug.exceptions import RequestTimeout

app= Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)