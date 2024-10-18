#models
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K

import data_utils as du
import keras_tuner as kt


def cnn(train_spectrogram_ds, input_shape,training=True):
    '''
    cnn model with normalization layer and resizing layer
    '''
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = Normalization()
    # Fit the state of the layer to the spectrograms
    if training:
        spec = train_spectrogram_ds.map(map_func=lambda spec, label: spec)
    
    norm_layer.adapt(data=train_spectrogram_ds)

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Resizing(120, 120))
    model.add(norm_layer)
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50))


    return model



def cnn_lstm(train_spectrogram_ds,input_shape):
    "same cnn+lstm model" 
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = Normalization()
    # Fit the state of the layer to the spectrograms
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Resizing(120, 120))
    model.add(norm_layer)

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Permute((2, 1, 3)))
    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, dropout=0.25, return_sequences=True)))

    model.add(Bidirectional(LSTM(128, dropout=0.25, return_sequences=False)))

    model.add(Dense(100))
    model.add(Dense(50))
    return model


class attention(Layer):
    '''
    Attention layer
    '''

    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
    


def cnn_lstm_attention(train_spectrogram_ds,input_shape):
    "same cnn_lstm + attention"
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Resizing(120, 120))
    model.add(norm_layer)

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Permute((2, 1, 3)))
    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, dropout=0.25, return_sequences=True)))
    model.add(attention(return_sequences=True))
    model.add(Bidirectional(LSTM(128, dropout=0.25, return_sequences=False)))
    model.add(Dense(100))
    model.add(Dense(50))
    return model




## The hypermodel
class cnn_tuner(kt.HyperModel):
    """
    Hypermodel for the CNN model
    """
    def __init__(self,train_spectrogram_ds, input_shape):
        self.input_shape = input_shape
        self.train_spectrogram_ds = train_spectrogram_ds
        
    def build(self, hp):
        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        norm_layer.adapt(data=self.train_spectrogram_ds.map(map_func=lambda spec, label: spec))

        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Resizing(120, 120))
        model.add(norm_layer)
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        #model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50))

        model.compile(
                optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
                )
        
        return model


def test_model(model_test, dataset_type):
    """
    Function to test a given model

    Args:
    model_test: model to test
    dataset_type: type of dataset to use

    Returns:
    None
    """

    augment=False # variable for image augmentation on the fly

    #set the data path and audio path
    if dataset_type == 'original':
        data_path = '/mnt/ESC-50-master/meta/esc50.csv'
        audio_path = '/mnt/ESC-50-master/audio_16k/'

    elif dataset_type == 'audio':
        data_path = "/mnt/ESC-50-master/audio_16k_aug_r4/esc50_augmented_r4.csv"
        audio_path = "/mnt/ESC-50-master/audio_16k_aug_r4/"


    elif dataset_type == 'img':
        data_path = '/mnt/ESC-50-master/meta/esc50.csv'
        audio_path = '/mnt/ESC-50-master/audio_16k/'
        augment=True

    #create a csv logger
    csv_logger = tf.keras.callbacks.CSVLogger(f'logs/{model_test.__name__}_training_16k_{dataset_type}.log',append=True)
    
    #load the metadata
    metadata=pd.read_csv(data_path)
    metadata["filename"] = metadata["filename"].apply(lambda x: audio_path + x)



    #iterate over the folds
    for fold in metadata["fold"].unique():
        #get the training and validation datasets
        train_df, val_df = du.get_fold(metadata, fold, batch_size=110*3, augment=augment)

        #get the input shape

        input_shape= du.get_shape(train_df)

        print("processing fold", fold)

        model = model_test(train_df,input_shape)
        

        model.compile(
                    optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'],
                    )
        
        #create other callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, 
        patience=15, verbose=0, mode='auto', baseline=None)

        #save the model with the best val accuracy
        save_model = tf.keras.callbacks.ModelCheckpoint(f'models/{model_test.__name__}_model_{dataset_type}_{fold}.keras', save_best_only=True)

        #train the model 
        history = model.fit(train_df, validation_data=val_df, epochs=100, callbacks=[early_stopping, csv_logger, save_model])

        # store best val accuracy
        best_val_acc = max(history.history['val_accuracy'])

        print("fold", fold, "best val accuracy", best_val_acc)

        
        del model
        K.clear_session()
        
        

       

    #average best val accuracy over all folds
    a="average best val accuracy", np.mean(best_val_acc), "for", dataset_type, "for", model_test.__name__
    #send the message to the telegram bot
    du.send_message(str(a))
    #write the message to a file
    f = open("big_log.txt", "a")
    #write in a new line
    f.write("\n")
    f.write(str(a))
    f.close()


