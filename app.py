
import re
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from flask import Flask, render_template, request, redirect, url_for, session
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()

twitter_data = pd.read_csv("Twitter_Data.csv")
twitter_data.dropna(inplace=True)
twitter_data.drop_duplicates(inplace=True)
assert 'clean_text' in twitter_data.columns, "'clean_text' column not found in dataset"

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    content = [port_stem.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

twitter_data['tweet'] = twitter_data['clean_text'].apply(stemming)

label_map = {0: "Neutral", 1: "Positive", -1: "Negative"}

X_train_text, X_test_text, Y_train, Y_test = train_test_split(
    twitter_data['tweet'], twitter_data['category'],
    test_size=0.2, stratify=twitter_data['category'], random_state=2
)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

pickle.dump(model, open("trained_model.sav", "wb"))
pickle.dump(vectorizer, open("vectorizer.sav", "wb"))


app = Flask(__name__)
app.secret_key = "supersecretkey"   


model = pickle.load(open("trained_model.sav", "rb"))
vectorizer = pickle.load(open("vectorizer.sav", "rb"))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().split()
    text = [word for word in text if word not in stop_words]
    text = [port_stem.stem(word) for word in text]
    return " ".join(text)


USERNAME = "admin"
PASSWORD = "1234"

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        name = request.form.get("name")   
        tweet = request.form.get("tweet") 
        if not tweet or tweet.strip()== "":
            error_msg="Please enter your text"
            return render_template("index.html",error=error_msg,name=name)

        cleaned = preprocess_text(tweet)
        vectorized_input = vectorizer.transform([cleaned])
        pred = int(model.predict(vectorized_input)[0])
        prediction = label_map[pred]

        return render_template("result.html",
                               name=name,
                               tweet=tweet,
                               prediction=prediction)
@app.route("/home")
def home():
    return index()

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid Credentials. Try again."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    
    counts = twitter_data["category"].value_counts().to_dict()

    
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        counts.get(1, 0),
        counts.get(0, 0),
        counts.get(-1, 0)
    ]

    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    chart_path = os.path.join("static", "sentiment_pie.png")
    plt.savefig(chart_path)
    plt.close()

    
    sample_texts = twitter_data.head(10).to_dict(orient="records")

    return render_template("dashboard.html",
                           counts=counts,
                           chart="sentiment_pie.png",
                           sample_texts=sample_texts)


if __name__ == "__main__":
    app.run(debug=True)
