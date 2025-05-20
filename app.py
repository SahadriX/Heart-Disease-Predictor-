from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

try:
    model = joblib.load("heart_disease_model.pkl")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            data = {
                "age": int(request.form["age"]),
                "sex": int(request.form["sex"]),
                "cp": int(request.form["cp"]),
                "trestbps": int(request.form["trestbps"]),
                "chol": int(request.form["chol"]),
            }
            df = pd.DataFrame([data])
            result = model.predict(df)[0]
            prediction = "⚠️ Likely Heart Disease" if result == 1 else "✅ No Heart Disease"
        except Exception as e:
            prediction = f"❌ Error: {e}"
    return render_template("index.html", prediction=prediction)
    
if __name__ == "__main__":
    app.run(debug=True)
