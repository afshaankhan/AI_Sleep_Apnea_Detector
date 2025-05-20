from flask import Flask, render_template, request, jsonify
import os
import json
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename
from wfdb_processor import convert_wfdb_to_model_input

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
HISTORY_PATH = os.path.join('static', 'history.json')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Ensure history file exists
if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'w') as f:
        json.dump([], f)

# Load model and scaler
with open('stacking_sleep_apnea_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/apnea-info')
def apnea_info():
    return render_template('apnea_info.html')

@app.route('/future')
def future():
    return render_template('future.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist("wfdb_files")
    if len(files) != 3:
        return jsonify({'error': 'Please upload .hea, .dat, and .st files'}), 400

    filenames = []
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        filenames.append(filename)

    patient_id = filenames[0].split('.')[0]

    try:
        X = convert_wfdb_to_model_input(patient_id, app.config['UPLOAD_FOLDER'], scaler_path='scaler.pkl')
        preds = model.predict(X)
        apnea_windows = int(preds.sum())
        total_windows = len(preds)
        has_apnea = apnea_windows / total_windows >= 0.05

        ahi_estimate = round((apnea_windows / total_windows) * 60, 2)  # AHI estimate (30s windows)

        result_text = '🚨 Apnea Detected' if has_apnea else '✅ No Apnea Detected'

        # Save result to history.json
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'result': result_text,
            'apnea_windows': apnea_windows,
            'total_windows': total_windows,
            'ahi': ahi_estimate,
            'details': preds[:30].tolist()  # Optional preview
        }

        with open(HISTORY_PATH, 'r') as f:
            history = json.load(f)

        history.insert(0, entry)
        history = history[:10]  # Keep only latest 10

        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

        return jsonify({
            'total_windows': total_windows,
            'apnea_windows': apnea_windows,
            'result': result_text,
            'details': preds.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        with open(HISTORY_PATH, 'w') as f:
            json.dump([], f)
        return jsonify({'status': 'cleared'})
    except:
        return jsonify({'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True)