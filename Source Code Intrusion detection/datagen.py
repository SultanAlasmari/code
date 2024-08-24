import pandas as pd
from missingpy import MissForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from Feature_Extraction import resnet50, extract_iot_features, statistical_features
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Model
from feature_selection import improved_mi_feature_selection
from save_load import save, load
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split


def datagen():

    data = pd.read_csv('./Dataset/DDoSdata.csv')
    data = data.drop(columns=['flgs', 'Unnamed: 0', 'proto', 'category', 'subcategory', 'saddr', 'daddr', 'state'])

    mixed_type_columns = data.select_dtypes(include=['object']).columns

    data = data.drop(columns=mixed_type_columns)

    data = data[:500000]

    # Plot histograms before preprocessing
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 3, 1)
    plt.hist(data['pkts'], bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of pkts')
    plt.xlabel('pkts')
    plt.ylabel('Frequency')
    plt.subplot(2, 3, 2)
    plt.hist(data['bytes'], bins=30, color='green', alpha=0.7)
    plt.title('Histogram of bytes')
    plt.xlabel('bytes')
    plt.ylabel('Frequency')
    plt.subplot(2, 3, 3)
    plt.hist(data['dur'], bins=30, color='red', alpha=0.7)
    plt.title('Histogram of dur')
    plt.xlabel('dur')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('./Data Visualization/Dataset pkts, bytes, dur visualisation.png')
    plt.show()

    # Preprocessing
    # Handling the missing values

    # Initialize MissForest imputer
    imputer = MissForest()

    # Impute missing values
    imputed_df = imputer.fit_transform(data)

    imputed_df = pd.DataFrame(imputed_df, columns=data.columns)

    label = imputed_df['attack']

    imputed_data = imputed_df.drop(columns=['attack'])

    plt.figure(figsize=(12, 8))

    columns_to_plot = ['pkts', 'bytes', 'dur', 'mean', 'stddev', 'sum']

    for i, col in enumerate(columns_to_plot, start=1):
        plt.subplot(2, 3, i)
        plt.hist(imputed_data[col], bins=30, color='green', alpha=0.7)
        plt.title(f'{col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('./Data Visualization/preprocessed.png')
    plt.show()

    # Calculate correlation matrix
    correlation_matrix = imputed_data.corr()

    # Plot heatmap
    plt.figure(figsize=(18, 14))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"fontsize": 6})
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(fontsize=5, rotation=50)
    plt.yticks(fontsize=6)
    plt.savefig('./Data Visualization/Correlation Matrix Heatmap.png')
    plt.show()

    # Outlier Detection - Local Outlier Factor (LOF)

    # Initialize the Local Outlier Factor model
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

    # The predicted labels: 1 for inliers, -1 for outliers
    outlier_labels = lof.fit_predict(imputed_data)

    outlier_scores = lof.negative_outlier_factor_

    imputed_data['outlier'] = outlier_labels

    imputed_data['outlier_score'] = outlier_scores

    imputed_data.to_csv('Dataset/preprocessed.csv', index=False)

    # Plotting before outlier detection
    plt.figure(figsize=(8, 6))
    plt.scatter(imputed_data['bytes'], imputed_data['pkts'], c='blue', label='Normal Data')
    # Marking outliers
    outliers_before = imputed_data[imputed_data['outlier'] == -1]
    plt.scatter(outliers_before['bytes'], outliers_before['pkts'], c='red', label='Outliers')
    plt.title('Outlier Detection')
    plt.xlabel('Bytes')
    plt.ylabel('Pkts')
    plt.legend()
    plt.savefig('./Data Visualization/Outlier Detected data.png')
    plt.show()

    imputed_data = pd.read_csv('./Dataset/preprocessed.csv')

    preprocessed_data = np.array(imputed_data)

    x_data = preprocessed_data.reshape(preprocessed_data.shape[0], preprocessed_data.shape[1], 1, 1)

    model = resnet50(input_shape=x_data[1].shape)

    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Deep learning based features
    resnet_features = feature_extractor.predict(x_data)

    # domain-based features
    iot_features = extract_iot_features(imputed_data)
    iot_features = np.array(iot_features)

    # statistical features
    stat_features = imputed_data.apply(statistical_features, axis=1)
    stat_features = np.array(stat_features)

    features = np.concatenate([resnet_features, iot_features, stat_features], axis =1)

    label = np.array(label)

    selected_features = improved_mi_feature_selection(features, label, k=100)

    # Absolute
    FEATURES = abs(selected_features)

    # Normalization
    FEATURES = FEATURES / np.max(FEATURES, axis=0)

    # Nan to Num Conversion
    FEATURES = np.nan_to_num(FEATURES)

    # Dimensionality Reduction - LocallyLinearEmbedding
    lle = LocallyLinearEmbedding(n_neighbors=2, n_components=10)

    batch_size = 1000

    # Initialize an empty array to store the reduced features
    X_reduced_all = np.empty((0, 10))  # n_components is the number of dimensions in the reduced space

    # Loop through the data in batches
    for i in range(0, len(FEATURES), batch_size):
        # Select a batch of data
        batch = FEATURES[i:i + batch_size]

        X_reduced_batch = lle.fit_transform(batch)

        # Append the reduced batch to the array
        X_reduced_all = np.vstack((X_reduced_all, X_reduced_batch))

    # Code for Visualizing the original Versus reduced data
    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.scatter(FEATURES[:, 0], FEATURES[:, 1], c=FEATURES[:, 2], cmap=plt.cm.Spectral)
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.subplot(122)
    plt.scatter(X_reduced_all[:, 0], X_reduced_all[:, 1], c=FEATURES[:, 2], cmap=plt.cm.Spectral)
    plt.title("Reduced Data (LLE)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.tight_layout()
    plt.savefig('Data Visualization/Dimensionality Reduced Features.png')
    plt.show()

    # training testing split
    train_sizes = [0.7, 0.8]
    for train_size in train_sizes:
        x_train, x_test, y_train, y_test = train_test_split(X_reduced_all, label, train_size=train_size)
        save('x_train_' + str(int(train_size * 100)), x_train)
        save('y_train_' + str(int(train_size * 100)), y_train)
        save('x_test_' + str(int(train_size * 100)), x_test)
        save('y_test_' + str(int(train_size * 100)), y_test)

