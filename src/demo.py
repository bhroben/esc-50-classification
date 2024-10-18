
import models as mo
import sounddevice as sd
from scipy.io.wavfile import write, read
import data_utils as du
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import time
import pandas as pd
import json
import seaborn as sns
import librosa

sns.set_style("whitegrid")

#import json file
def load_classes():
    """Load the class indices from the JSON file.
    
    Returns:
        dict: A dictionary containing the class indices.
    """

    with open('categories.json', 'r') as f:
        class_indices = json.load(f)
    return class_indices




fs = 16000  # Sample rate
seconds = 5  # Duration of recording

def inference(model,spec):
    """Perform inference on the given spectrogram using the provided model.

    Args:
        model (tf.keras.Model): The model to perform inference with.
        spec (tf.Tensor): The input spectrogram.

    Returns:
        int: The predicted class index.
        float: The time taken for inference.
    """
    start = time.time()
    pred = model(spec)
    end = time.time()
    y_pred = np.argmax(pred)

    return y_pred, end-start

def load_audio(audio='output.wav'):
    """Load the audio file and return the audio tensor.

    Args:
        audio (str): The path to the audio file.

    Returns:
        tf.Tensor: The audio tensor.
    """

    if type(audio) == str: 
        data,fs = librosa.load(audio, sr=16000)
    else:
        fs = 16000
        data = audio

    myrecording = tf.convert_to_tensor(data, dtype=tf.float32)

    #return error if audio is not 5 seconds
    if myrecording.shape[0] >= 80000:
        #trim to 5 seconds
        myrecording = myrecording[:80000]
    elif myrecording.shape[0] < 80000:
        #pad to 5 seconds
        myrecording = tf.pad(myrecording, [[0, 80000-myrecording.shape[0]]])

    if fs != 16000:
        return 'Error: Sample rate must be 16000'

    return myrecording

def make_spec(myrecording):
    """Create a mel spectrogram from the given audio tensor.

    Args:
        myrecording (tf.Tensor): The audio tensor.

    Returns:
        tf.Tensor: The mel spectrogram.
    """

    spec = du.mel_layer(myrecording)
    #add batch dimension
    spec = tf.expand_dims(spec, axis=0)

    return spec

def  plot_spec(spec):
    """Plot the mel spectrogram.

    Args:
        spec (tf.Tensor): The mel spectrogram.
    """

  
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec[0,:,:,0].numpy(), sr=16000, x_axis='time', y_axis='mel',n_fft=2048, hop_length=512, fmin=0, fmax=8000)
    plt.title('Mel spectrogram')
    plt.tight_layout()
    return plt


def classify_spec(spectrogram,runs=10):
    """Classify the given spectrogram using the CNN, LSTM, and Attention models.
    
    Args:
        spectrogram (tf.Tensor): The input spectrogram.
        runs (int): The number of times to run each model.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions and time values for each model.
    """

    # Create a DataFrame to store predictions and time values for 10 runs
    columns = ['Prediction'] + [f'Time_Run_{i+1}' for i in range(10)]
    df = pd.DataFrame(index=['CNN', 'LSTM', 'Attention'], columns=columns)

    # Load class indices
    class_indices = load_classes()

    # Load models
    cnn = tf.keras.models.load_model('models/cnn_model_audio_4.keras')
    lstm = tf.keras.models.load_model('models/cnn_lstm_model_audio_4.keras')
    attention = tf.keras.models.load_model('models/cnn_lstm_attention_model_audio_4.keras', custom_objects={'Attention': mo.Attention})

    # Run each model 10 times and store results
    for model_name, model in zip(['CNN', 'LSTM', 'Attention'], [cnn, lstm, attention]):
        times = []
        prediction = None

        for i in range(runs):
            pred, time_taken = inference(model, spectrogram)

            # Store the prediction from the first run (assuming predictions don't change)
            if prediction is None:
                prediction = pred

            # Store time for the current run
            df.loc[model_name, f'Time_Run_{i+1}'] = round(time_taken * 1000, 3)  # Convert to ms and round

        # Store the prediction in the 'Prediction' column
        df.loc[model_name, 'Prediction'] = class_indices[str(int(prediction))]

    return df

def plot_bar(df):
    """Plot a bar graph showing the mean and standard deviation of the time taken by each model.
    
    Args:
        df (pd.DataFrame): A DataFrame containing the predictions and time values for each model.

    Returns:
        plt.Figure: The bar graph.
    """
    
    time_columns = [f'Time_Run_{i+1}' for i in range(10)]
    df['Time_Mean'] = df[time_columns].mean(axis=1)
    df['Time_Std'] = df[time_columns].std(axis=1)

    fig, ax = plt.subplots()

    # Create a bar plot with error bars
    ax.bar(df.index, df['Time_Mean'], yerr=df['Time_Std'], color="skyblue", capsize=5)

    # Set labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Time Taken by Different Models (Mean and Std Dev)')

    # Adding the prediction labels on top of each bar
    for i, (model, row) in enumerate(df.iterrows()):
        ax.text(i, row['Time_Mean'] + row['Time_Std'] + 2, row['Prediction'], ha='center')

    ax.set_ylim(0, df['Time_Mean'].max() + df['Time_Std'].max() + 20)

    return fig