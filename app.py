from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)


with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form["review"]
        cleaned = clean_text(user_text)
        vect_text = vectorizer.transform([cleaned])
        sentiment = model.predict(vect_text)[0]

    return render_template(
        "index.html",
        sentiment=sentiment,
        user_text=user_text
    )

if __name__ == "__main__":
    app.run(debug=True)
