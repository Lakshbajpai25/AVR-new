from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/contactus')
def contactus():
    return render_template("contactus.html")


@app.route('/blog')
def blog():
    return render_template("blog.html")



@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)