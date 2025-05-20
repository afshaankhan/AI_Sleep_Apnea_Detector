import wfdb
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft
import pickle

def spectral_energy(signal):
    fft_vals = rfft(signal)
    return np.sum(np.abs(fft_vals)**2) / len(fft_vals)

def spectral_entropy(signal):
    fft_vals = rfft(signal)
    psd = np.abs(fft_vals)**2
    psd_norm = psd / (np.sum(psd) + 1e-7)
    return -np.sum(psd_norm * np.log(psd_norm + 1e-7))

def extract_features_from_signal(signal):
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'median': np.median(signal),
        'range': np.ptp(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal),
        'rms': np.sqrt(np.mean(np.square(signal))),
        'spectral_energy': spectral_energy(signal),
        'spectral_entropy': spectral_entropy(signal)
    }

def extract_features(window):
    feature_vector = {}
    for key, signal in window.items():
        if key != 'Label':
            feats = extract_features_from_signal(signal)
            for feat_name, value in feats.items():
                feature_vector[f"{key}_{feat_name}"] = value
    return feature_vector

def process_patient_data(patient_id, data_dir):
    file_path = f'/{patient_id}'
    record = wfdb.rdheader(data_dir + file_path)
    signals, fields = wfdb.rdsamp(data_dir + file_path)
    annotations = wfdb.rdann(data_dir + file_path, 'st')

    signal_names = fields['sig_name']
    signals_to_keep = ['ECG', 'BP', 'EEG']
    sampling_rate = 250
    window_length_samples = 30 * sampling_rate

    start_index = annotations.sample[0]
    adjusted_start_index = int((start_index / sampling_rate) * sampling_rate)

    filtered_signals = {
        name: signals[:, i][adjusted_start_index:]
        for i, name in enumerate(signal_names)
        if any(sig in name for sig in signals_to_keep)
    }

    windowed_signals = {
        name: signal[:(len(signal) // window_length_samples) * window_length_samples].reshape(-1, window_length_samples)
        for name, signal in filtered_signals.items()
    }

    apnea_values = ['H', 'HA', 'OA', 'X', 'CA', 'CAA']
    apnea_labels = [1 if any(marker in note for marker in apnea_values) else 0 for note in annotations.aux_note]

    return {
        f"{patient_id}_Window_{i}": {
            **{n: w[i] for n, w in windowed_signals.items() if i < len(w)},
            'Label': apnea_labels[i]
        }
        for i in range(min(len(next(iter(windowed_signals.values()))), len(apnea_labels)))
    }

def convert_wfdb_to_model_input(patient_id, data_dir, scaler_path="scaler.pkl"):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    windows = process_patient_data(patient_id, data_dir)
    features = []

    for window in windows.values():
        feats = extract_features(window)
        df = pd.DataFrame([feats])
        df.fillna(df.mean(), inplace=True)

        expected_columns = scaler.feature_names_in_
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]

        X_scaled = scaler.transform(df)
        features.append(X_scaled[0])

    return np.array(features)