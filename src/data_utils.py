import numpy as np
import pandas as pd
import librosa
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from audiomentations import AddGaussianSNR,PitchShift, Compose,AddGaussianNoise,TimeStretch,Shift
import requests

#path to the original audio files
audio_path_orig="/mnt/ESC-50-master/audio_16k/"

#image augmentation with tensorflow keras layers
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomBrightness(0.1),
  tf.keras.layers.RandomTranslation(0.1, 0.1),
  tf.keras.layers.RandomContrast(0.4)])




def get_fold(df, fold_num, batch_size=128,augment=False,shuffle=True):
    """
    Function to get the fold i for validation and the rest for training

    Args:
    df: pandas dataframe with the audio files and their labels
    fold_num: fold number to use for validation
    batch_size: batch size for the dataset
    augment: boolean to augment the data
    shuffle: boolean to shuffle the data

    Returns:
    train_spectrogram_ds: dataset with the training data
    val_spectrogram_ds: dataset with the validation data
    """

    #select fold i for validation and the rest for train
    train_df = df[df["fold"]!=fold_num]

    # take the fold i for validation from the original dataset with no augmentation

    df_orig = pd.read_csv('/mnt/ESC-50-master/meta/esc50.csv')
    df_orig["filename"] = df_orig["filename"].apply(lambda x: audio_path_orig + x)

    val_df = df_orig[df_orig["fold"]==fold_num]


    
    
    X_train = np.array(train_df["filename"].values )
    y_train = np.array(train_df["target"].values)

    #create a dataset from the filenames and labels
    train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    #map the filenames to the audio files
    train_loader = train_loader.map(lambda x, y: (tf.audio.decode_wav(tf.io.read_file(x), desired_channels=1)[0], y))


    #squeeze the audio and make spectrogram
    train_loader = train_loader.map(squeeze, tf.data.AUTOTUNE)
    train_spectrogram_ds = make_spec_ds(train_loader)

    if augment:
        #apply data augmentation
        train_spectrogram_ds = train_spectrogram_ds.map(lambda x,y:(data_augmentation(x),y), num_parallel_calls=tf.data.AUTOTUNE)
        train_spectrogram_ds=train_spectrogram_ds.repeat(5)

    if shuffle:
        #shuffle the dataset, len(train_df) is the number of samples to shuffle
        train_spectrogram_ds = train_spectrogram_ds.shuffle(len(train_df))

    #batch the dataset
    train_spectrogram_ds = train_spectrogram_ds.batch(batch_size)

    #same for validation
    X_val = np.array(val_df["filename"].values )
    y_val = np.array(val_df["target"].values)

    val_loader = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_loader = val_loader.map(lambda x, y: (tf.audio.decode_wav(tf.io.read_file(x), desired_channels=1)[0], y))


    #squeeze the audio and make spectrogram
    val_loader = val_loader.map(squeeze, tf.data.AUTOTUNE)
    val_spectrogram_ds = make_spec_ds(val_loader)

    if shuffle:
        val_spectrogram_ds = val_spectrogram_ds.shuffle(len(train_df))

    val_spectrogram_ds=val_spectrogram_ds.batch(batch_size)#

    return train_spectrogram_ds, val_spectrogram_ds

def get_shape(train_spectrogram_ds):
    """
    Function to get the shape of the input for the model

    Args:
    train_spectrogram_ds: dataset with the training data

    Returns:
    input_shape: shape of the input for the model
    """
    #the dataset is an iterator
    it = iter(train_spectrogram_ds)
    example=next(it)
    input_shape = example[0].shape  
    #remove the batch dimension
    input_shape=input_shape[1:]

    return input_shape


def load_data(audio_path_train, audio_path_val):
    """
    ### This function is DEPRECATED, use the function get_fold instead ###
    Function to load the data from the audio files and create the dataset using audio_dataset_from_directory
    this requires the audio files to be in the following structure in the directories, one for training and one for validation:
    audio_path
    ├── class_1
    │   ├── file_1.wav
    │   ├── file_2.wav
    │   └── ...
    ├── class_2
    │   ├── file_1.wav
    │   ├── file_2.wav
    │   └── ...
    └── ...

    
    Args:
    audio_path_train: path to the training audio files
    audio_path_val: path to the validation audio files


    Returns:
    train_ds: dataset with the training data
    val_ds: dataset with the validation data
    """


    train_ds= tf.keras.utils.audio_dataset_from_directory(
        directory=audio_path_train,
        batch_size=256,
        validation_split=None,
        seed=0,
        sampling_rate=None,
        output_sequence_length=None,
        subset=None)

    val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=audio_path_val,
        batch_size=256,
        validation_split=None,
        seed=0,
        sampling_rate=None,
        output_sequence_length=None,
        subset=None)

    label_names = np.array(train_ds.class_names)

    # check if labels are loaded correctly
    if len(label_names) != 50:
       raise ValueError("Expected 50 labels, but got ", len(label_names))
    
    #squeeze the audio

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    #convert audio to mel spectrogram
    train_ds = make_spec_ds(train_ds)
    val_ds = make_spec_ds(val_ds)

    #cache the dataset
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def squeeze(audio, labels):
  """Squeeze the audio with shape (batch, time, 1) to (batch, time).
  Args:
    audio: Tensor of audio with shape (batch, time, 1).
    labels: Tensor of labels.
    Returns:
    Squeezed audio and labels.
    """
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def mel_layer(x):
    """
    Function to create the mel spectrogram layer
    Args:
    x: input tensor
    Returns:
    spectrogram: mel spectrogram
    """

    #create mel spectrogram
    x=tf.keras.layers.MelSpectrogram(
        fft_length=2048,
        sequence_stride=512, 
        sequence_length=None,
        window='hann', #hann or hamming
        sampling_rate=16000,
        num_mel_bins=128,
        min_freq=20.0,
        max_freq=None,
        power_to_db=True,
        top_db=80.0,
        mag_exp=2.0,
        min_power=1e-10,
        ref_power=1.0,

    )(x)
    #add channel dimension for CNN
    spectrogram = x[..., tf.newaxis]
    
    return spectrogram

def make_spec_ds(ds):
    """
    Function to create the mel spectrogram dataset
    Args:
    ds: dataset with the audio files
    Returns:
    ds: dataset with the mel spectrograms
    """

    return ds.map(
      map_func=lambda audio,label: (mel_layer(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)


def mel_info(train_ds):
    """
    Function to plot the mel spectrogram of a random sample from the dataset
    Args:
    train_ds: dataset with the training data
    """

    for spectrogram, label in train_ds.take(1):
        tensor=tf.squeeze(spectrogram)
        print('Mel shape:', spectrogram.shape)
        
        print('Mel spectrogram:')
        #extract random number from the batch
        tot= len(tensor)
        print('Total number of samples:', tot)
        idx = np.random.randint(0, tot)
        print('Label:', label[idx].numpy())
        librosa.display.specshow(tensor[idx].numpy(), sr=16000, x_axis='time', y_axis='mel',)



def plot_confusion_matrix(val_df, model, metadata):
    """
    Function to plot the confusion matrix
    Args:
    val_df: validation dataset
    model: trained model
    metadata: metadata with the classes
    """

    #predict the classes
    y_pred = np.argmax(model.predict(val_df), axis=-1)

    #extract classes from val_df to use in confusion matrix
    y_val = tf.concat([y for x, y in val_df], axis=0)


    #sort metadata by target
    classes=metadata.sort_values(by=["target"])
    classes_names=classes["category"].unique()


    # map classes to their names in y_val and y_pred
    y_val = [classes_names[i] for i in y_val]
    y_pred = [classes_names[i] for i in y_pred]

    #plot confusion matrix
    cn_matrix = confusion_matrix(y_val,y_pred,normalize='true')
    plt.figure(figsize = (20,20))
    sns.heatmap(cn_matrix, annot=True, xticklabels=list(classes_names), yticklabels=list(classes_names))

    plt.savefig('imgs/confusion_matrix.png')


def send_message(message):
    """
    Function to send a message to a telegram chat, used for sending validation results
    https://stackoverflow.com/questions/75116947/how-to-send-messages-to-telegram-using-python

    Args:
    message: message to send
    """
    TOKEN = "123456789:ABCDEF" #token of the bot
    chat_id = "123456789" #chat id
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json()