from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import json
import subprocess
import os

app = Flask(__name__)


def load_results():
    """Return parsed results.json or a default dict if missing/malformed."""
    try:
        with open("static/results.json") as f:
            return json.load(f)
    except Exception:
        return {
            "best_model": "N/A",
            "r2_score": None,
            "rmse": None,
            "mae": None,
            "message": "Results not available. Run training."
        }


@app.route('/results')
def results():
    return jsonify(load_results())


@app.route('/train', methods=['GET', 'POST'])
def train():
    """Trigger training by running main.py in background. Returns immediately."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(project_dir, 'main.py')

    if not os.path.exists(main_py):
        return jsonify({"status": "error", "message": "main.py not found"}), 404

    # Ensure static dir exists
    os.makedirs(os.path.join(project_dir, 'static'), exist_ok=True)

    lock_path = os.path.join(project_dir, 'static', 'training.lock')
    log_path = os.path.join(project_dir, 'static', 'train.log')

    # If lock exists, do not start another training
    if os.path.exists(lock_path):
        return jsonify({"status": "running", "message": "Training is already running"}), 409

    # Build a shell command that creates a lock, runs main.py and writes logs, then removes the lock
    # Use 'exec' so the PID belongs to the python process; start_new_session detaches it from this process
    # Build a safe shell command string with single-quoted paths to avoid nested-quote issues
    cmd = f"bash -lc \"set -o pipefail; touch '{lock_path}'; python3 '{main_py}' >> '{log_path}' 2>&1; rm -f '{lock_path}'\""

    try:
        subprocess.Popen(cmd, cwd=project_dir, shell=True, start_new_session=True)
        return jsonify({"status": "started", "message": "Training started in background"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/train_status')
def train_status():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    lock_path = os.path.join(project_dir, 'static', 'training.lock')
    log_path = os.path.join(project_dir, 'static', 'train.log')

    running = os.path.exists(lock_path)

    # Read last 5000 characters of the log (or whole file if smaller)
    tail = ""
    try:
        if os.path.exists(log_path):
            with open(log_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                # read last ~5000 bytes
                to_read = min(size, 5000)
                f.seek(size - to_read)
                tail = f.read().decode('utf-8', errors='ignore')
    except Exception:
        tail = "(could not read log)"

    return jsonify({"running": running, "log": tail})


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}

    try:
        features = [
            float(data.get('RH2M', 0)),
            float(data.get('GWETTOP', 0)),
            float(data.get('QV2M', 0)),
            float(data.get('ALLSKY_SFC_SW_DWN', 0)),
            float(data.get('T2MDEW', 0)),
            float(data.get('T2M_MIN', 0)),
            float(data.get('DOY', 0)),
            float(data.get('WS2M', 0))
        ]
    except ValueError:
        return jsonify({"error": "Invalid input; all feature values must be numeric."}), 400

    model_path = 'rainfall_model.pkl'
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found. Please run training first."}), 404

    try:
        # Load model per-request so latest trained model is used without restarting the server.
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict([features])[0]
        prediction = max(prediction, 0)

        return jsonify({
            "prediction": round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)