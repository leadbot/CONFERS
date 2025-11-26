from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle

def train_autoencoder_classifier(X_train, y_train, label_encoder, 
                                 autoencoder_epochs=15, 
                                 classifier_epochs=100,
                                 activation='relu',
                                 dropout_rate=0.2,
                                 layer_size=128,
                                 num_layers=2,
                                 learning_rate=0.001,
                                 batch_size=32,
                                 plot=False):
    
    #Opt
    optimizer = Adam(learning_rate=learning_rate)

    #Input
    input_layer = Input(shape=(X_train.shape[1],))
    x = input_layer

    #encoder
    for _ in range(num_layers):
        x = Dense(layer_size)(x)
        x = Activation(activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    #Bottlenec
    bottleneck = Dense(24)(x)

    #Decoder
    x = bottleneck
    for _ in range(num_layers):
        x = Dense(layer_size)(x)
        x = Activation(activation)(x)

    decoded_output = Dense(X_train.shape[1], activation='linear')(x)  # Use linear for reconstruction

    #Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded_output)
    autoencoder.compile(optimizer=optimizer, loss='mae')

    #Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    #Train autoencoder
    history_autoencoder = autoencoder.fit(
        X_train, X_train,
        epochs=autoencoder_epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[reduce_lr, early_stopping],
        verbose=0
    )

    #Encoder model for classification
    encoder = Model(inputs=input_layer, outputs=bottleneck)
    encoded_input = encoder.output

    #Classifier
    x = Dense(layer_size)(encoded_input)
    x = Dropout(dropout_rate)(x)
    x = Dense(32)(x)
    classifier_output = Dense(len(label_encoder.classes_), activation='softmax')(x)

    ###Full model
    model_autoencoder = Model(inputs=input_layer, outputs=classifier_output)
    model_autoencoder.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    ##Train classifier
    history_classifier = model_autoencoder.fit(
        X_train, y_train,
        epochs=classifier_epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    #Pedict on test data
    y_pred_probs = model_autoencoder.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    #compute F1 score on test data
    f1 = f1_score(y_test, y_pred, average='macro')

    #Plot
    if plot:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history_autoencoder.history['loss'], label='Autoencoder Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Autoencoder Training')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history_classifier.history['loss'], label='Train Loss')
        plt.plot(history_classifier.history['val_loss'], label='Val Loss')
        plt.plot(history_classifier.history['accuracy'], label='Train Acc')
        plt.plot(history_classifier.history['val_accuracy'], label='Val Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Loss / Accuracy')
        plt.title('Classifier Training')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return model_autoencoder, history_autoencoder, history_classifier, f1, y_test, y_pred