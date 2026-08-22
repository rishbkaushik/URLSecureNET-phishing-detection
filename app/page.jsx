"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  const [url, setUrl] = useState("");
  const [model, setModel] = useState("logistic");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    setError("");

    try {
      // Create form data for Flask
      const formData = new FormData();

      formData.append("url", url);
      formData.append("model", model);

      // Send URL to Flask backend
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      // if (!response.ok) {
      //   throw new Error("Flask backend returned an error");
      // }
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Flask error:", errorText);
        throw new Error(`Flask backend returned ${response.status}: ${errorText}`);
      }

      const data = await response.json();

      console.log("Flask response:", data);

      // Send prediction information to result page
      const params = new URLSearchParams();

      params.set("url", data.url);
      params.set("prediction", String(data.final_prediction));

      // Send model results
      params.set("results", JSON.stringify(data.results));

      // Send confidence values
      params.set("confidence", JSON.stringify(data.confidence));

      router.push(`/result?${params.toString()}`);

      } catch (error) {
         console.error("ACTUAL ERROR:", error);
         setError(error.message);
        }
    // } catch (error) {
    //   console.error("Error:", error);

    //   setError(
    //     "Unable to connect to Flask backend. Make sure Flask is running on port 5000."
    //   );
     finally {
      setLoading(false);
    }
  };

  return (
    <div className="wrapper">

      <header>

        <img
          className="logo-big"
          src="/logo1.png"
          alt="URLSecureNET logo"
        />

        <h1>URLSecureNET</h1>

        <p>
          AI-Powered URL Security & Phishing Detection System
        </p>

      </header>

      <main>

        <div className="card">

          <h2>Analyze URL Safety</h2>

          <form onSubmit={handleSubmit}>

            <label>
              Enter URL
            </label>

            <input
              type="text"
              name="url"
              placeholder="https://example.com"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              required
            />

            <label>
              Select Model
            </label>

            <select
              name="model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >

              <option value="logistic">
                Logistic Regression
              </option>

              <option value="nb">
                Naive Bayes
              </option>

              <option value="rf">
                Random Forest
              </option>

              <option value="svm">
                SVM
              </option>

            </select>

            <button
              type="submit"
              disabled={loading}
            >

              {loading ? "⏳ Analyzing..." : "🔍 Analyze"}

            </button>

          </form>

          {error && (
            <p
              style={{
                color: "red",
                marginTop: "15px",
                textAlign: "center",
              }}
            >
              {error}
            </p>
          )}

        </div>

      </main>

      <footer>
        © 2026 URLSecureNET | Secure Browsing with Machine Learning
      </footer>

    </div>
  );
}