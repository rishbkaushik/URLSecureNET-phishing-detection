from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# =========================
# LOAD MODELS (ONLY ONCE)
# =========================
log_model = pickle.load(open("models/logistic_phishing.pkl", "rb"))
nb_model = pickle.load(open("models/Naive_Bayes_phishing.pkl", "rb"))
# rf_model = pickle.load(open("models/Random_forest_phishing.pkl", "rb"))
svm_model = pickle.load(open("models/svm_model.pkl", "rb"))

# Load vectorizer once (IMPORTANT 🚀)
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# =========================
# HOME ROUTE
# =========================
@app.route('/')
def home():
    return render_template("index.html")

# =========================
# PREDICT ROUTE
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    url = request.form['url']

    # 🔥 CLEAN URL
    cleaned_url = re.sub(r'^https?://(www\.)?', '', url).lower()

    # Convert to vector
    vector_input = vectorizer.transform([cleaned_url])

    # =========================
    # RUN ALL MODELS
    # =========================
    results = {}
    confidence = {}

    # Logistic
    pred_log = log_model.predict(vector_input)[0]
    results['Logistic Regression'] = pred_log
    if hasattr(log_model, "predict_proba"):
        confidence['Logistic Regression'] = round(max(log_model.predict_proba(vector_input)[0]) * 100, 2)

    # Naive Bayes
    pred_nb = nb_model.predict(vector_input)[0]
    results['Naive Bayes'] = pred_nb
    confidence['Naive Bayes'] = round(max(nb_model.predict_proba(vector_input)[0]) * 100, 2)

    # Random Forest
    # pred_rf = rf_model.predict(vector_input)[0]
    # results['Random Forest'] = pred_rf
    # confidence['Random Forest'] = round(max(rf_model.predict_proba(vector_input)[0]) * 100, 2)

    # SVM (may not support probability)
    pred_svm = svm_model.predict(vector_input)[0]
    results['SVM'] = pred_svm

    # =========================
    # FINAL DECISION (MAJORITY)
    # =========================
    votes = list(results.values())
    final_prediction = max(set(votes), key=votes.count)

    # =========================
    # PASS TO FRONTEND
    # =========================
    return render_template(
        "result.html",
        url=url,
        results=results,
        confidence=confidence,
        final_prediction=final_prediction
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)