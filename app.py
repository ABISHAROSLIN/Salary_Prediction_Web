from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("salary_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        experience = float(request.form["experience"])

        prediction = model.predict([[experience]])[0]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)