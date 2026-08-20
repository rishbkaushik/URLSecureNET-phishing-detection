from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import os

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODELS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

log_model = pickle.load(
    open(os.path.join(MODEL_DIR, "logistic_phishing.pkl"), "rb")
)

nb_model = pickle.load(
    open(os.path.join(MODEL_DIR, "Naive_Bayes_phishing.pkl"), "rb")
)

svm_model = pickle.load(
    open(os.path.join(MODEL_DIR, "svm_model.pkl"), "rb")
)

vectorizer = pickle.load(
    open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb")
)


# =========================
# HOME ROUTE
# =========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "URLSecureNET Flask backend is running"
    })


# =========================
# PREDICT ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get URL from Next.js FormData
        url = request.form.get("url")

        if not url:
            return jsonify({
                "error": "URL is required"
            }), 400

        # =========================
        # CLEAN URL
        # =========================

        cleaned_url = re.sub(
            r"^https?://(www\.)?",
            "",
            url
        ).lower()

        # =========================
        # VECTORIZE
        # =========================

        vector_input = vectorizer.transform([cleaned_url])

        # =========================
        # RUN MODELS
        # =========================

        results = {}
        confidence = {}

        # Logistic Regression
        pred_log = log_model.predict(vector_input)[0]

        results["Logistic Regression"] = str(pred_log)

        if hasattr(log_model, "predict_proba"):
            confidence["Logistic Regression"] = round(
                max(log_model.predict_proba(vector_input)[0]) * 100,
                2
            )

        # Naive Bayes
        pred_nb = nb_model.predict(vector_input)[0]

        results["Naive Bayes"] = str(pred_nb)

        if hasattr(nb_model, "predict_proba"):
            confidence["Naive Bayes"] = round(
                max(nb_model.predict_proba(vector_input)[0]) * 100,
                2
            )

        # SVM
        pred_svm = svm_model.predict(vector_input)[0]

        results["SVM"] = str(pred_svm)

        # =========================
        # FINAL DECISION
        # =========================

        votes = list(results.values())

        final_prediction = max(
            set(votes),
            key=votes.count
        )

        # =========================
        # RETURN JSON TO NEXT.JS
        # =========================

        return jsonify({
            "url": url,
            "final_prediction": final_prediction,
            "results": results,
            "confidence": confidence
        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )