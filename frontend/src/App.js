import React, { useState, useEffect } from "react";

// Because frontend/package.json has a "proxy" pointing at the backend,
// we can call "/api/..." directly - no full http://localhost:5000 needed,
// and no CORS setup required anywhere.
const PREDICT_URL = "/api/predict";
const HISTORY_URL = "/api/history";

function formatDateGroup(isoTimestamp) {
  const date = new Date(isoTimestamp);
  const now = new Date();
  const startOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate());

  const diffDays = Math.round((startOfDay(now) - startOfDay(date)) / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function groupHistoryByDate(history) {
  const groups = {};
  history.forEach((item) => {
    const key = formatDateGroup(item.timestamp);
    if (!groups[key]) groups[key] = [];
    groups[key].push(item);
  });
  return groups;
}

function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  // Load past scans (today, yesterday, older) as soon as the page opens.
  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await fetch(HISTORY_URL);
      const data = await response.json();
      setHistory(data);
    } catch (err) {
      // If history can't load, the rest of the app should still work.
      console.error("Could not load history:", err);
    }
  };

  const checkUrl = async () => {
    if (!url.trim()) {
      setError("Please enter a URL first.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(PREDICT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Server returned an error.");
      }

      setResult(data);
      setHistory((prev) => [data, ...prev]); // update instantly...
      loadHistory(); // ...then re-sync with what's actually saved on disk
    } catch (err) {
      setError(
        err.message.includes("Python")
          ? err.message
          : "Could not reach the backend server. Make sure 'python server.py' is running in the backend folder."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") checkUrl();
  };

  const groupedHistory = groupHistoryByDate(history);

  return (
    <div className="page">
      <div className="container">
        <h1>🛡️ Phishing URL &amp; Malicious Link Classifier</h1>
        <p className="subtitle">
          Machine Learning based detector — paste any URL below to check
          whether it looks safe or suspicious, and see exactly why.
        </p>

        <div className="input-row">
          <input
            type="text"
            placeholder="e.g. http://paypal-verify-login.xyz/confirm"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button onClick={checkUrl} disabled={loading}>
            {loading ? "Checking..." : "Check URL"}
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <div
            className={`result-card ${
              result.is_phishing ? "result-danger" : "result-safe"
            }`}
          >
            <h2>{result.is_phishing ? "⚠️ Phishing / Malicious" : "✅ Legitimate"}</h2>
            <p className="checked-url">{result.url}</p>

            <div className="confidence-bar-wrapper">
              <div className="confidence-labels">
                <span>Legitimate: {result.probability_legitimate}%</span>
                <span>Phishing: {result.probability_phishing}%</span>
              </div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${result.probability_phishing}%` }}
                />
              </div>
            </div>

            {/* THE "WHY" SECTION */}
            {result.reasons && result.reasons.length > 0 && (
              <div className="reasons-box">
                <h3>
                  Why is this {result.is_phishing ? "flagged as Phishing" : "considered Legitimate"}?
                </h3>
                <ul className="reasons-list">
                  {result.reasons.map((reason, i) => (
                    <li key={i} className={result.is_phishing ? "reason-danger" : "reason-safe"}>
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <details>
              <summary>See raw extracted feature values</summary>
              <table className="feature-table">
                <tbody>
                  {Object.entries(result.features).map(([key, value]) => (
                    <tr key={key}>
                      <td>{key}</td>
                      <td>{String(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </div>
        )}

        {Object.keys(groupedHistory).length > 0 && (
          <div className="history">
            <h3>Recent checks</h3>
            {Object.entries(groupedHistory).map(([dateLabel, items]) => (
              <div key={dateLabel} className="history-group">
                <div className="history-date-label">{dateLabel}</div>
                <ul>
                  {items.map((h, i) => (
                    <li key={i} className={h.is_phishing ? "hist-danger" : "hist-safe"}>
                      <span>{h.url}</span>
                      <span>{h.label} ({h.confidence}%)</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}

        <footer>
          Built with React (frontend) + Python http.server (backend) + Scikit-learn (ML model)
        </footer>
      </div>
    </div>
  );
}

export default App;
