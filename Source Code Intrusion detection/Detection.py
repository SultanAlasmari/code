from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Attention, concatenate
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,matthews_corrcoef, cohen_kappa_score, brier_score_loss


def metrices(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f_measure = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    bsl = brier_score_loss(y_test, y_pred)
    met = [accuracy, precision, recall, f_measure, mcc, bsl]
    return met


# Define CNN model
def create_cnn_model(input_shape):
    cnn_input = Input(shape=input_shape)
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_input)
    maxpool1 = MaxPooling1D(pool_size=2)(conv1)
    cnn_output = Dense(128, activation='relu')(maxpool1)
    cnn_model = Model(inputs=cnn_input, outputs=cnn_output)
    return cnn_model


# Define LSTM model
def create_lstm_model(input_shape):
    lstm_input = Input(shape=input_shape)
    lstm_layer = LSTM(64, return_sequences=False)(lstm_input)
    lstm_output = Dense(128, activation='relu')(lstm_layer)
    lstm_model = Model(inputs=lstm_input, outputs=lstm_output)
    return lstm_model


# Define Autoencoder model
def create_autoencoder_model(input_shape):
    autoencoder_input = Input(shape=input_shape)
    encoded = Dense(128, activation='relu')(autoencoder_input)
    decoded = Dense(10, activation='sigmoid')(encoded)
    autoencoder_output = Dense(128)(decoded)
    autoencoder_model = Model(inputs=autoencoder_input, outputs=autoencoder_output)
    return autoencoder_model


# Define combined model
def Ensemble_Net(x_train, y_train, x_test, y_test, epochs, batch_size):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    num_classes = int(max(y_test))

    cnn_model = create_cnn_model(x_train[1].shape)
    lstm_model = create_lstm_model(x_train[1].shape)
    autoencoder_model = create_autoencoder_model(x_train[1].shape)

    cnn_input = cnn_model.input
    lstm_input = lstm_model.input
    autoencoder_input = autoencoder_model.input

    cnn_output = cnn_model.output
    lstm_output = lstm_model.output
    autoencoder_output = autoencoder_model.output
    attn_layer = Attention()([cnn_output, autoencoder_output])

    flat = Flatten()(attn_layer)
    # Concatenate CNN, LSTM and Autoencoder outputs
    combined_output = concatenate([flat, lstm_output])

    # Final classification layer
    output = Dense(num_classes, activation='sigmoid')(combined_output)

    combined_model = Model(inputs=[cnn_input, lstm_input, autoencoder_input], outputs=output)

    combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = combined_model.fit([x_train, x_train, x_train], y_train, epochs=epochs, batch_size=batch_size, validation_data=([x_test, x_test, x_test], y_test))

    pred = combined_model.predict([x_test, x_test, x_test])
    binary_pred = np.mean((pred >= 0.5).astype(int), axis=1)

    met = metrices(y_test, binary_pred)
    return pred, met, history


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU,SimpleRNN

def DNN(x_train, y_train, x_test, y_test):
    num_classes = len(set(y_test))

    model = Sequential()
    model.add(Dense(128, input_shape=x_train[1].shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=1)
    pred = np.argmax(model.predict(x_test), axis=1)
    return pred, metrices(y_test, pred)


def gru(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    num_classes = len(set(y_test))

    model = Sequential([
        GRU(units=64, return_sequences=True, input_shape=x_train[1].shape),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # train the model
    model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=0)

    y_predict = np.argmax(model.predict(x_test), axis=1)
    return y_predict, metrices(y_test, y_predict)


def RNN(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    num_classes = len(set(y_test))

    model = Sequential()
    model.add(SimpleRNN(64, input_shape=x_train[1].shape))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=0)
    pred = np.argmax(model.predict(x_test), axis=1)
    met = metrices(y_test, pred)
    return pred, met


def DBN(x_train, y_train, x_test, y_test):

    # Define the Deep Belief Network architecture
    dbn = Sequential()
    # First hidden layer
    dbn.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    # Second hidden layer
    dbn.add(Dense(64, activation='relu'))
    # Third hidden layer
    dbn.add(Dense(32, activation='relu'))
    # Output layer
    dbn.add(Dense(1, activation='sigmoid'))
    # Compile the model
    dbn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    dbn.fit(x_train, y_train, epochs=100, batch_size=32)

    y_pred = dbn.predict(x_test)
    y_pred = np.round(np.mean(y_pred, axis=1))
    met = metrices(y_test, y_pred)
    return y_pred, met
