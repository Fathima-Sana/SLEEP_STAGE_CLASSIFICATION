from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import numpy as np
from scipy.signal import butter, lfilter
from keras.models import load_model
from collections import Counter

app = Flask(__name__)


# Define the folder where NPZ files are stored
NPZ_FOLDER = 'NPZ'

# Define the high-pass filter function
def butter_bandpass_filter(data, highpass, fs, order=4):
    nyq = 0.5 * fs
    high = highpass / nyq
    b, a = butter(order, high, btype='highpass')
    y = lfilter(b, a, data)
    return y

# Load the NPZ file containing EEG data for the subject
def load_subject_data(npz_file):
    data = np.load(npz_file)
    X = data['x']
    return X

# Preprocess the data (apply high-pass filter)
def preprocess_data(X):
    highpass_cutoff = 40.0  # Define the high-pass cutoff frequency
    fs = 100  # Sampling frequency
    preprocessed_data = np.array([butter_bandpass_filter(sample, highpass=highpass_cutoff, fs=fs, order=4) for sample in X])
    return preprocessed_data

# Predict sleep stages for the subject
def predict_subject_sleep_stages(subject_data, model):
    preprocessed_data = preprocess_data(subject_data)
    predictions = model.predict(preprocessed_data)
    sleep_stages = np.argmax(predictions, axis=1)  # Get the index of the maximum probability as the predicted sleep stage
    return sleep_stages

@app.route('/attn_sleep_model', methods=['POST'])
def attn_sleep_model():
    model_file = 'AttnSleep_model.h5'  # Filename of the AttnSleep model file

    # Get the filename from the request
    npz_filename = request.json.get('npz_filename')

    # Construct the full path to the NPZ file
    npz_file = os.path.join(NPZ_FOLDER, npz_filename)
    if not os.path.isfile(npz_file):
        return jsonify({'error': 'NPZ file not found'})

    # Assume these functions exist and do what's expected
    subject_data = load_subject_data(npz_file)
    model = load_model(model_file)
    if model is None:
        return jsonify({'error': 'Model loading failed'})

    predicted_sleep_stages = predict_subject_sleep_stages(subject_data, model)
    label_mapping = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4'
    }

    # Convert numerical labels to sleep stage names
    predicted_sleep_stages_names = [label_mapping[label] for label in predicted_sleep_stages]
    print(predicted_sleep_stages_names)
    # Return predicted sleep stages as JSON response
    return jsonify({'predicted_sleep_stages': predicted_sleep_stages_names})

@app.route('/result')
def result():
    # Render the template with the predicted sleep stages
    return render_template('result.html')

@app.route('/deepsleep_sleep_model', methods=['POST'])
def deepsleep_sleep_model():
    model_file = 'DeepSleep_model.h5'  # Filename of the AttnSleep model file
   # Get the filename from the request
    npz_filename = request.json.get('npz_filename')

    # Construct the full path to the NPZ file
    npz_file = os.path.join(NPZ_FOLDER, npz_filename)
    if not os.path.isfile(npz_file):
        return jsonify({'error': 'NPZ file not found'})

    # Assume these functions exist and do what's expected
    subject_data = load_subject_data(npz_file)
    model = load_model(model_file)
    if model is None:
        return jsonify({'error': 'Model loading failed'})

    predicted_sleep_stages = predict_subject_sleep_stages(subject_data, model)
    label_mapping = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4'
    }

    # Convert numerical labels to sleep stage names
    predicted_sleep_stages_names = [label_mapping[label] for label in predicted_sleep_stages]
    print(predicted_sleep_stages_names)
    # Return predicted sleep stages as JSON response
    return jsonify({'predicted_sleep_stages': predicted_sleep_stages_names})

@app.route('/eeg_sleep_model', methods=['POST'])
def eeg_sleep_model():
    model_file = 'SleepEEG_model.h5'  

   # Get the filename from the request
    npz_filename = request.json.get('npz_filename')

    # Construct the full path to the NPZ file
    npz_file = os.path.join(NPZ_FOLDER, npz_filename)
    if not os.path.isfile(npz_file):
        return jsonify({'error': 'NPZ file not found'})

    # Assume these functions exist and do what's expected
    subject_data = load_subject_data(npz_file)
    model = load_model(model_file)
    if model is None:
        return jsonify({'error': 'Model loading failed'})

    predicted_sleep_stages = predict_subject_sleep_stages(subject_data, model)
    label_mapping = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4'
    }

    # Convert numerical labels to sleep stage names
    predicted_sleep_stages_names = [label_mapping[label] for label in predicted_sleep_stages]
    print(predicted_sleep_stages_names)
    # Return predicted sleep stages as JSON response
    return jsonify({'predicted_sleep_stages': predicted_sleep_stages_names})


@app.route('/model_selection.html')
def model_selection():
    return render_template('model_selection.html')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/selectedf.html')
def selectedf():
    return render_template('selectedf.html')

@app.route('/go_to_index')
def go_to_index():
    return redirect(url_for('index'))


@app.route('/npz_process')
def npz_process():
    npz_filename = request.args.get('npz_filename')
    x_values_first_10 = request.args.get('x_values_first_10')
    return render_template('npz_process.html', npz_filename=npz_filename, x_values_first_10=x_values_first_10)



@app.route('/preprocess', methods=['POST'])
def preprocess():
    if 'filename' not in request.files:
        return jsonify({'error': 'No file chosen'})

    file = request.files['filename']

    if file.filename == '':
        return jsonify({'error': 'No file chosen'})

    if not file.filename.lower().endswith('.edf'):
        return jsonify({'error': 'File should be in EDF format'})

    # Get the filename from the FileStorage object
    edf_filename = file.filename

    # Construct NPZ filename by replacing "-PSG.edf" with ".npz"
    npz_filename = os.path.join(edf_filename.replace('-PSG.edf', '.npz'))
    npz_file_path = os.path.join(NPZ_FOLDER, npz_filename)
    
    # Check if the NPZ file exists
    if not os.path.isfile(npz_file_path):
        return jsonify({'error': 'Try Another File'})

    # Load NPZ data
    npz_data = np.load(npz_file_path, allow_pickle=True)
    
    # Extract 'x' values from NPZ data
    x_values = npz_data['x'][:1]
    x_values_first_10 = x_values[:, :60]
    
    # Return the NPZ filename and x_values_first_10 in the JSON response
    return jsonify({'npz_filename': npz_filename, 'x_values_first_10': x_values_first_10.tolist()})

@app.route('/model_selection_summary', methods=['POST'])
def model_selection_summary():
    # Get the filename from the request
    npz_filename = request.json.get('npz_filename')

    # Construct the full path to the NPZ file
    npz_file = os.path.join(NPZ_FOLDER, npz_filename)
    if not os.path.isfile(npz_file):
        return jsonify({'error': 'NPZ file not found'})

    subject_data = load_subject_data(npz_file)
    attn_model = load_model('AttnSleep_model.h5')
    deepsleep_model = load_model('DeepSleep_model.h5')
    eeg_model = load_model('SleepEEG_model.h5')

    if attn_model is None or deepsleep_model is None or eeg_model is None:
        return jsonify({'error': 'Model loading failed'})

    attn_predicted_sleep_stages = predict_subject_sleep_stages(subject_data, attn_model)
    deepsleep_predicted_sleep_stages = predict_subject_sleep_stages(subject_data, deepsleep_model)
    eeg_predicted_sleep_stages = predict_subject_sleep_stages(subject_data, eeg_model)

    # Count the occurrences of each sleep stage in each model's predictions
    attn_sleep_stage_counts = Counter(attn_predicted_sleep_stages)
    deepsleep_sleep_stage_counts = Counter(deepsleep_predicted_sleep_stages)
    eeg_sleep_stage_counts = Counter(eeg_predicted_sleep_stages)

    # Total number of predictions for each model
    attn_total_predictions = len(attn_predicted_sleep_stages)
    deepsleep_total_predictions = len(deepsleep_predicted_sleep_stages)
    eeg_total_predictions = len(eeg_predicted_sleep_stages)

    # Compute the percentage of each sleep stage for each model
    attn_sleep_stage_percentages = {stage: (count / attn_total_predictions) * 100 for stage, count in attn_sleep_stage_counts.items()}
    deepsleep_sleep_stage_percentages = {stage: (count / deepsleep_total_predictions) * 100 for stage, count in deepsleep_sleep_stage_counts.items()}
    eeg_sleep_stage_percentages = {stage: (count / eeg_total_predictions) * 100 for stage, count in eeg_sleep_stage_counts.items()}
    # Render the template with the predicted sleep stages from all models and return it as a response
    return render_template('/ModelSummary.html', 
                           attn_sleep_stage_percentages=attn_sleep_stage_percentages,
                           deepsleep_sleep_stage_percentages=deepsleep_sleep_stage_percentages,
                           eeg_sleep_stage_percentages=eeg_sleep_stage_percentages)


if __name__ == '__main__':
    app.run(debug=True)
