import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model
from scipy.stats import skew, kurtosis
import pandas as pd


# ResNet - Residual neural network

def identity_block(x, filters, kernel_size=3, strides=1):
    shortcut = x

    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def resnet_conv_block(x, filters, kernel_size=3, strides=2):
    shortcut = x

    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def resnet50(input_shape):

    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Max pooling
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual blocks
    x = resnet_conv_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)

    x = resnet_conv_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)

    x = resnet_conv_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)

    x = resnet_conv_block(x, 512)
    x = identity_block(x, 512)
    x = identity_block(x, 512)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    # Fully connected layer
    x = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='resnet50_non_image')
    return model


def extract_iot_features(data):

    data['is_tcp'] = (data['proto_number'] == 6).astype(int)
    data['is_udp'] = (data['proto_number'] == 17).astype(int)
    data['is_icmp'] = (data['proto_number'] == 1).astype(int)

    data['is_http_server'] = ((data['AR_P_Proto_P_Dport'] == 80) | (data['AR_P_Proto_P_Sport'] == 80)).astype(
        int)
    data['is_https_server'] = ((data['AR_P_Proto_P_Dport'] == 443) | (data['AR_P_Proto_P_Sport'] == 443)).astype(
        int)
    data['is_dns_server'] = ((data['AR_P_Proto_P_Dport'] == 53) | (data['AR_P_Proto_P_Sport'] == 53)).astype(
        int)
    return data


def statistical_features(row):
    return pd.Series({
        'mean': row.mean(),
        'median': row.median(),
        'std': row.std(),
        'min': row.min(),
        'max': row.max(),
        'skew': skew(row),
        'kurtosis': kurtosis(row)
    })