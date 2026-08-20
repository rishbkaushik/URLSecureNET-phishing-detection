"use client";

import { useSearchParams } from "next/navigation";

export default function ResultPage() {
  const searchParams = useSearchParams();

  const url = searchParams.get("url") || "No URL provided";
  const prediction = searchParams.get("prediction") || "safe";

  const results = JSON.parse(
    searchParams.get("results") || "{}"
  );

  const confidence = JSON.parse(
    searchParams.get("confidence") || "{}"
  );

  const isBad =
    String(prediction).toLowerCase() === "bad" ||
    String(prediction).toLowerCase() === "phishing";

  return (
    <div className="wrapper">

      <header>

        <h1>🔐 URLSecureNET</h1>

        <p>
          Secure Your Browsing with Smart Phishing Detection
        </p>

      </header>

      <main>

        <div className="card result-card">

          <h2>🔍 Analysis Result</h2>

          <div className="url-box">

            <p>Analyzed URL</p>

            <span>{url}</span>

          </div>

          <div className="result-box">

            {isBad ? (

              <div className="status bad">
                ❌ Phishing Website Detected
              </div>

            ) : (

              <div className="status good">
                ✅ Safe Website
              </div>

            )}

          </div>

          <h3>📊 Model Comparison</h3>

          <table>

            <thead>

              <tr>
                <th>Model</th>
                <th>Prediction</th>
                <th>Confidence</th>
              </tr>

            </thead>

            <tbody>

              <tr>
                <td>Logistic Regression</td>

                <td>
                  {results["Logistic Regression"] ?? "N/A"}
                </td>

                <td>
                  {confidence["Logistic Regression"] != null
                    ? `${confidence["Logistic Regression"]}%`
                    : "N/A"}
                </td>
              </tr>

              <tr>
                <td>Naive Bayes</td>

                <td>
                  {results["Naive Bayes"] ?? "N/A"}
                </td>

                <td>
                  {confidence["Naive Bayes"] != null
                    ? `${confidence["Naive Bayes"]}%`
                    : "N/A"}
                </td>
              </tr>

              <tr>
                <td>Random Forest</td>

                <td>
                  {results["Random Forest"] ?? "N/A"}
                </td>

                <td>
                  {confidence["Random Forest"] != null
                    ? `${confidence["Random Forest"]}%`
                    : "N/A"}
                </td>
              </tr>

              <tr>
                <td>SVM</td>

                <td>
                  {results["SVM"] ?? "N/A"}
                </td>

                <td>
                  {confidence["SVM"] != null
                    ? `${confidence["SVM"]}%`
                    : "N/A"}
                </td>
              </tr>

            </tbody>

          </table>

          <a href="/" className="back-btn">
            ← Analyze Another URL
          </a>

        </div>

      </main>

      <footer>
        © 2026 URLSecureNET
      </footer>

    </div>
  );
}