import os

from flask import Flask, jsonify, render_template_string, request

from FraudDetector import FraudDetector


def resolve_model_dir() -> str:
    env_model_dir = os.environ.get("FRAUD_MODEL_DIR", "").strip()
    if env_model_dir:
        return env_model_dir

    candidates = sorted(
        [item for item in os.listdir(".") if item.startswith("fraud_call_model_") and os.path.isdir(item)],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No trained model directory found. Train the model first or set FRAUD_MODEL_DIR.")
    return candidates[0]


MODEL_DIR = resolve_model_dir()
detector = FraudDetector(MODEL_DIR)
app = Flask(__name__)


INDEX_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fraud Call Analyzer</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: #fffaf2;
      --ink: #1e1a16;
      --accent: #1f6f78;
      --accent-2: #d96c3f;
      --border: #d9ccb8;
      --ok: #2d7d46;
      --bad: #b63c31;
      --warn: #b17a16;
      --muted: #6f655c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, #f8d9b8 0, transparent 22%),
        radial-gradient(circle at left bottom, #d6ebe4 0, transparent 24%),
        var(--bg);
    }
    .wrap {
      max-width: 980px;
      margin: 32px auto;
      padding: 24px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 18px 60px rgba(80, 57, 36, 0.08);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 42px;
      line-height: 1;
    }
    p {
      color: var(--muted);
      margin: 0 0 20px;
      font-size: 18px;
    }
    textarea {
      width: 100%;
      min-height: 240px;
      resize: vertical;
      border-radius: 16px;
      border: 1px solid var(--border);
      padding: 16px;
      font: inherit;
      font-size: 16px;
      background: #fff;
      color: var(--ink);
    }
    .row {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 16px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 14px 22px;
      font-size: 16px;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), #2d8d97);
      color: white;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.05); }
    .hint {
      font-size: 14px;
      color: var(--muted);
    }
    .result {
      margin-top: 24px;
      display: none;
      border-top: 1px solid var(--border);
      padding-top: 20px;
    }
    .pill {
      display: inline-block;
      padding: 8px 14px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 15px;
      margin-right: 8px;
    }
    .fraud { background: #fde2df; color: var(--bad); }
    .normal { background: #def3e3; color: var(--ok); }
    .suspicious { background: #fff0cf; color: var(--warn); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 16px;
    }
    .box {
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fff;
    }
    .label {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: var(--muted);
    }
    .value {
      margin-top: 6px;
      font-size: 18px;
      font-weight: 700;
      word-break: break-word;
    }
    ul {
      margin: 10px 0 0;
      padding-left: 20px;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 14px;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin-top: 14px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Fraud Call Analyzer</h1>
      <p>Вставь транскрипт звонка и страница покажет, считает ли модель разговор мошенническим.</p>

      <textarea id="transcript" placeholder="Например: Здравствуйте я звоню из банка..."></textarea>

      <div class="row">
        <button id="analyzeBtn">Анализировать</button>
        <span class="hint">Модель: {{ model_dir }}</span>
      </div>

      <div class="result" id="result">
        <div id="headline"></div>
        <div class="grid">
          <div class="box">
            <div class="label">Fraud Probability</div>
            <div class="value" id="fraudProbability">-</div>
          </div>
          <div class="box">
            <div class="label">Risk Level</div>
            <div class="value" id="riskLevel">-</div>
          </div>
          <div class="box">
            <div class="label">Scenario</div>
            <div class="value" id="scenarioType">-</div>
          </div>
          <div class="box">
            <div class="label">Channel</div>
            <div class="value" id="channel">-</div>
          </div>
        </div>

        <div class="box" style="margin-top:16px;">
          <div class="label">Markers</div>
          <ul id="markers"></ul>
        </div>

        <div class="box" style="margin-top:16px;">
          <div class="label">Recommendation</div>
          <div class="value" id="recommendation" style="font-size:16px; font-weight:500;"></div>
        </div>

        <div class="box" style="margin-top:16px;">
          <div class="label">Почему модель так решила</div>
          <ul id="reasons"></ul>
        </div>

        <div class="box" style="margin-top:16px;">
          <div class="label">Top Suspicious Segment</div>
          <pre id="segmentText">-</pre>
        </div>
      </div>
    </div>
  </div>

  <script>
    const btn = document.getElementById("analyzeBtn");
    const transcriptEl = document.getElementById("transcript");
    const resultEl = document.getElementById("result");

    btn.addEventListener("click", async () => {
      const transcript = transcriptEl.value.trim();
      if (!transcript) {
        alert("Вставь транскрипт звонка.");
        return;
      }

      btn.disabled = true;
      btn.textContent = "Анализ...";

      try {
        const response = await fetch("/analyze-call", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transcript })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Ошибка сервера");
        }

        const predictedClass = data.predicted_class || "unknown";
        const pillClass = predictedClass === "fraud"
          ? "fraud"
          : (predictedClass === "suspicious" ? "suspicious" : "normal");
        document.getElementById("headline").innerHTML =
          `<span class="pill ${pillClass}">${predictedClass.toUpperCase()}</span>`;
        document.getElementById("fraudProbability").textContent =
          `${((data.fraud_probability || 0) * 100).toFixed(2)}%`;
        document.getElementById("riskLevel").textContent = data.risk_level || "-";
        document.getElementById("scenarioType").textContent = data.features?.scenario_type || "-";
        document.getElementById("channel").textContent = data.features?.channel || "-";
        document.getElementById("recommendation").textContent = data.recommendation || "-";

        const markersEl = document.getElementById("markers");
        markersEl.innerHTML = "";
        (data.markers || []).forEach(marker => {
          const li = document.createElement("li");
          li.textContent = marker;
          markersEl.appendChild(li);
        });

        const reasonsEl = document.getElementById("reasons");
        reasonsEl.innerHTML = "";
        (data.decision_reasons || []).forEach(reason => {
          const li = document.createElement("li");
          li.textContent = reason;
          reasonsEl.appendChild(li);
        });

        const topSegment = (data.suspicious_segments && data.suspicious_segments[0]?.text) || "Подозрительный сегмент не найден";
        document.getElementById("segmentText").textContent = topSegment;
        resultEl.style.display = "block";
      } catch (error) {
        alert(error.message);
      } finally {
        btn.disabled = false;
        btn.textContent = "Анализировать";
      }
    });
  </script>
</body>
</html>
"""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_dir": MODEL_DIR})


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, model_dir=MODEL_DIR)


@app.route("/analyze-call", methods=["GET", "POST"])
def analyze_call():
    if request.method == "GET":
        return render_template_string(INDEX_HTML, model_dir=MODEL_DIR)

    payload = request.get_json(force=True)
    transcript = payload.get("transcript", "")
    window_size = int(payload.get("window_size", 3))
    step = int(payload.get("step", 1))

    if not transcript.strip():
        return jsonify({"error": "transcript is required"}), 400

    result = detector.analyze_call(transcript, window_size=window_size, step=step)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
