# PTB-XL ECG Classifier

This project involves creating a classifier for the PTB-XL dataset to determine whether an ECG record is normal or abnormal.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#Requirements)
- [Data Preprocessing](#Data-Preprocessing)
- [CNN Model Training](#Model-Training)
- [CNN Model Evaluation](#Model-Evaluation)
- [CRNN Model Training](#Model-Training)
- [CRNN Model Evaluation](#Model-Evaluation)
- [Results](#results)


## Introduction
The PTB-XL dataset is a large collection of electrocardiography (ECG) records. This project aims to build a classifier to identify whether a given ECG record is normal or abnormal using a Convolutional Recurrent Neural Network (CRNN) model.

## Dataset
The PTB-XL dataset is available at [PhysioNet](https://physionet.org/content/ptb-xl/1.0.1/). It contains 21,837 clinical 12-lead ECG records from 18,885 patients, each lasting 10 seconds and annotated with up to 71 different ECG statements.


## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- wfdb
- scikit-learn
- imbalanced-learn (imblearn)
- TensorFlow
- Keras
  
 You can install the necessary packages using pip:
```bash
   pip install pandas numpy matplotlib wfdb scikit-learn imbalanced-learn tensorflow keras
 ```
## Data Preprocessing
# Step 1: Load and Filter the Dataset
Load the dataset and filter out records with noise or electrode problems.

```python
import pandas as pd

# Load the CSV file into a DataFrame
file_path = "ptbxl_database.csv"
data = pd.read_csv(file_path)

# Filter the dataset
filtered_data = data[(data['burst_noise'].isnull()) & 
                     (data['electrodes_problems'].isnull()) & 
                     (data['extra_beats'].isnull())]

# Filter for records containing 'NORM' and 'IMI' in 'scp_codes' column
norm_data = filtered_data[filtered_data['scp_codes'].str.contains('NORM')]
imi_data = filtered_data[filtered_data['scp_codes'].str.contains('IMI')]

# Balance the dataset
min_records = min(norm_data.shape[0], imi_data.shape[0])
norm_data = norm_data.sample(n=min_records, random_state=42)
imi_data = imi_data.sample(n=min_records, random_state=42)

# Concatenate the two subsets
final_data = pd.concat([norm_data, imi_data])

# Save the concatenated data to a new CSV file
final_data.to_csv('New_data.csv', index=False)
```
# Step 2: Label the Data
Extract and label the relevant classes from the dataset
```python
def collect_and_label(dataset):
    df = pd.read_csv(dataset)
    alpha = df['scp_codes'].str.split("'").str[1].str[-2:] == 'IMI'
    beta = df['scp_codes'].str.split("'").str[1] == 'NORM'
    df = df[alpha | beta]
    df['label'] = df['scp_codes'].str.split("'").str[1]
    return df

df_labeled = collect_and_label('New_data.csv')
```
# Step 3: Balance and Augment the Data
Use SMOTE to balance and augment the dataset.

```python
from imblearn.over_sampling import SMOTENC

def balance_and_augment(df):
    smote_nc = SMOTENC(categorical_features=[1], random_state=0)
    X_res, y_res = smote_nc.fit_resample(df[['ecg_id', 'filename_hr']].to_numpy(), df['label'])
    df_balanced = pd.DataFrame(X_res, columns=['ecg_id', 'filename_hr'])
    df_balanced['label'] = y_res
    return df_balanced

df_balanced_and_augmented = balance_and_augment(df_labeled)
```
# Step 4: Apply High-Pass Filter
Apply a Butterworth high-pass filter to the ECG signals.
```python
import numpy as np
from scipy.signal import butter, filtfilt
import wfdb

def apply_highpass_filter(signal, lowcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='high')
    return filtfilt(b, a, signal)

def apply_highpass_filter_to_lead1(file_path):
    record = wfdb.rdrecord(file_path)
    lead_1_signal = record.p_signal[:, 0]
    fs = record.fs
    lowcut = 0.5
    filtered_lead_1_signal = apply_highpass_filter(lead_1_signal, lowcut, fs)
    return filtered_lead_1_signal

file_paths = df_balanced_and_augmented['filename_hr']
filtered_signals = [apply_highpass_filter_to_lead1(file_path) for file_path in file_paths]
df_balanced_and_augmented['filtered_lead_1_channel_0'] = filtered_signals

```
# Step 5: Split the Data
Split the data into training, validation, and test sets.
```python
from sklearn.model_selection import train_test_split

train, val_test = train_test_split(df_balanced_and_augmented, train_size=0.7, random_state=1002)
validation, test = train_test_split(val_test, test_size=0.5, random_state=1002)
```
# Step 6: Preprocess the Data
```python
Prepare the data for model training.
def preprocess(dat):
    data_dir = list(dat['filename_hr'])
    data_signal = map(read_signal, data_dir)
    data_signal = list(data_signal)
    data_signal = np.array(data_signal)
    data_dict = {'NORM': 0, 'IMI': 1}
    encoded_label = dat['label'].map(data_dict)
    return np.array(data_signal), np.array(encoded_label)

def read_signal(record):
    tes = wfdb.rdrecord(record, sampfrom=0, sampto=5000)
    signal = tes.__dict__['p_signal'][:, 0]
    return signal

X_train, y_train = preprocess(train)
X_valid, y_valid = preprocess(validation)
X_test, y_test = preprocess(test)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
```
## CNN Model Training
# CNN Model
Define and train a Convolutional Neural Network (CNN) model.
```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint

def create_model(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

model = create_model(input_shape=X_train[0].shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_valid, y_valid), callbacks=[checkpoint])
  ```
## CNN Model Evaluation
Evaluate the model using the test set.
 ```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

y_pred_proba = model.predict(X_test)
```

# Threshold Optimization
Find the best threshold for binary classification.
```python
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, f1_score

thresholds = np.linspace(0, 1, 1000)
best_threshold = None
best_metric_value = float('-inf')

for threshold in thresholds:
    y_pred_thresholded = (y_pred_proba > threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_thresholded)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    metric_value = TP + TN - FP - FN
    if metric_value > best_metric_value:
        best_metric_value = metric_value
        best_threshold = threshold

print("Best Threshold:", best_threshold)
print("Best Metric Value:", best_metric_value)

y_pred_optimal = (y_pred_proba > best_threshold).astype(int)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
TP_optimal = cm_optimal[1, 1]
TN_optimal = cm_optimal[0, 0]
FP_optimal = cm_optimal[0, 1]
FN_optimal = cm_optimal[1, 0]

precision_optimal = precision_score(y_test, y_pred_optimal)
f1_optimal = f1_score(y_test, y_pred_optimal)
sensitivity_optimal = TP_optimal / (TP_optimal + FN_optimal)
specificity_optimal = TN_optimal / (TN_optimal + FP_optimal)

print("Precision with Optimal Threshold:", precision_optimal)
print("F1 Score with Optimal Threshold:", f1_optimal)
print("Sensitivity (Recall) with Optimal Threshold:", sensitivity_optimal)
print("Specificity with Optimal Threshold:", specificity_optimal)
```
# ROC Curve
Plot the ROC curve and mark the optimal threshold.
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

optimal_threshold_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds_roc[optimal_threshold_index]
plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], c='red', marker='o', label='Optimal Threshold')
plt.legend()
plt.show()
```
# Confusion Matrix
Plot the confusion matrix for the optimal threshold.
```python
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

classes = ['Negative', 'Positive']
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cm_optimal, classes)
plt.show()
```

## CRNN Model Training
# CRNN Model
Define and train a Convolutional Recurrent Neural Network (CRNN) model.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint

# Function to build the CRNN model
def build_crnn_model(input_shape, num_classes):
    model = Sequential()
    
    # CNN layers
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # LSTM layer
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Flatten before dense layer
    model.add(Flatten())
    
    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model
# Build the CRNN model
crnn_model = build_crnn_model(input_shape, num_classes)

# Compile the model with a lower learning rate
crnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])

# Print model summary
crnn_model.summary()

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_sparse_categorical_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
batch_size = 32
epochs = 20  # Increased number of epochs

history = crnn_model.fit(X_train, y_train,  batch_size=batch_size,  epochs=epochs,  validation_data=(X_valid, y_valid), callbacks=[checkpoint])

```

## CRNN Model Evaluation

```python
# Evaluate the model on the test set
test_loss, test_accuracy = crnn_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

```
# Threshold Optimization
```python
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, f1_score

# Define the threshold search range
thresholds = np.linspace(0, 1, 1000)

# Initialize variables to store best threshold and corresponding metrics
best_threshold = None
best_metric_sum = float('-inf')  # We want to maximize this sum

# Loop through each threshold and calculate corresponding metrics
for threshold in thresholds:
    y_pred_thresholded = (y_pred_proba[:, 1] > threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_thresholded)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    # Calculate the sum of true positives and true negatives while minimizing false positives and false negatives
    metric_sum = TP + TN - FP - FN
    
    # Update best threshold and best metric sum if a better threshold is found
    if metric_sum > best_metric_sum:
        best_metric_sum = metric_sum
        best_threshold = threshold

# Print the best threshold and corresponding metrics
print("Best Threshold:", best_threshold)
print("Best Metric Sum (TP + TN - FP - FN):", best_metric_sum)

# Convert predicted probabilities to class labels using the optimal threshold
y_pred_optimal = (y_pred_proba[:, 1] > best_threshold).astype(int)

# Compute confusion matrix using the optimal threshold
cm_optimal = confusion_matrix(y_test, y_pred_optimal)

# Extract TP, TN, FP, FN using the optimal threshold
TP_optimal = cm_optimal[1, 1]
TN_optimal = cm_optimal[0, 0]
FP_optimal = cm_optimal[0, 1]
FN_optimal = cm_optimal[1, 0]

# Print the number of TP, TN, FP, FN using the optimal threshold
print("True Positives with Optimal Threshold:", TP_optimal)
print("True Negatives with Optimal Threshold:", TN_optimal)
print("False Positives with Optimal Threshold:", FP_optimal)
print("False Negatives with Optimal Threshold:", FN_optimal)

# Compute precision, F1 score, sensitivity (recall), and specificity
precision_optimal = precision_score(y_test, y_pred_optimal)
f1_optimal = f1_score(y_test, y_pred_optimal)
sensitivity_optimal = TP_optimal / (TP_optimal + FN_optimal)
specificity_optimal = TN_optimal / (TN_optimal + FP_optimal)

# Print precision, F1 score, sensitivity (recall), and specificity
print("Precision with Optimal Threshold:", precision_optimal)
print("F1 Score with Optimal Threshold:", f1_optimal)
print("Sensitivity (Recall) with Optimal Threshold:", sensitivity_optimal)
print("Specificity with Optimal Threshold:", specificity_optimal)
```
# ROC Curve
Plot the ROC curve and mark the optimal threshold.
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming you have already computed y_pred_proba using your CRNN model
y_pred_proba = crnn_model.predict(X_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba[:, 1])  # Assuming y_test is binary (0 or 1)

roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Mark optimal threshold point on ROC curve
optimal_threshold_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds_roc[optimal_threshold_index]
plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], c='red', marker='o', label='Optimal Threshold')
plt.legend()

plt.show()
```
# Confusion Matrix
Plot the confusion matrix for the optimal threshold.
```python
from sklearn.metrics import confusion_matrix, precision_score, f1_score
import matplotlib.pyplot as plt
import itertools

# Define function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Assuming you have already computed the optimal threshold metrics
best_threshold = 0.8188188188188188
y_pred_optimal = (y_pred_proba[:, 1] > best_threshold).astype(int)  # Assuming y_pred_proba is available

# Compute confusion matrix using the optimal threshold
cm_optimal = confusion_matrix(y_test, y_pred_optimal)

# Define class labels
classes = ['Negative', 'Positive']  # Replace with your actual class labels

# Print confusion matrix details
print("Confusion Matrix with Optimal Threshold:")
print(cm_optimal)

# Plot confusion matrix for optimal threshold
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cm_optimal, classes)
plt.show()

```
## Usage
1. Prepare the dataset by downloading it from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.1/) and placing it in the appropriate directory.
2. Run the provided script to process the dataset and balance the classes:
    ```bash
    python process_dataset.py
    ```
3. Train the model:
    ```bash
    python train.py
    ```
4. Evaluate the model:
    ```bash
    python evaluate.py
    ```
5. Predict using the model:
    ```bash
    python predict.py --input path_to_ecg_record
    ```

## Results
The CRNN model achieves an accuracy of 71% on the test set. and The CNN model achieves an accuracy of 70.7%
Detailed performance metrics and plots are generated during evaluation.
