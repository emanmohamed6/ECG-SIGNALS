#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb 
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import StratifiedKFold
import pandas as pd


# In[2]:


datasetdf_path = 'New data.csv'
# Read dataset into a DataFrame
datasetdf_df = pd.read_csv(datasetdf_path)

# Display the first few rows of the DataFrame
print(datasetdf_df.head())


# #Code

# ###Code
Code
# In[3]:


def collect_and_label(dataset):
    # Collect only the MI classes and the NORM classes from the dataset.
    df = pd.read_csv(dataset)
    alpha = df['scp_codes'].str.split("'").str[1].str[-2:] == 'MI'  # Collect all the MI classes.
    beta = df['scp_codes'].str.split("'").str[1] == 'NORM'  # Collect all the Normal classes.
    df = df[alpha | beta]
    df['label'] = df['scp_codes'].str.split("'").str[1]  # Create a new column 'label' containing categorical labels.

    return df


# In[4]:


df_labeled = collect_and_label('New data.csv')
print(df_labeled.head())


# In[5]:


label_counts = Counter(df_labeled['label'])
for label, count in label_counts.items():
    print(f"Label '{label}' has {count} records.")


# In[6]:


plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Frequency of Each Class in the Dataset')
plt.show()


# In[7]:


# Assuming your dataset is stored in a DataFrame called 'df'
# and the label column is named 'label'

# Filter the data to include only records with labels 'NORM' and 'IMI'
data = df_labeled[df_labeled['label'].isin(['NORM', 'IMI'])]

# Display the first few rows of the filtered data to verify
print(data.head())


# In[8]:


label_counts = Counter(data['label'])
for label, count in label_counts.items():
    print(f"Label '{label}' has {count} records.")


# In[9]:


from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC

def balance_and_augment(df):
    # Augment the dataset
    smote_nc = SMOTENC(categorical_features=[1], random_state=0)
    X_res, y_res = smote_nc.fit_resample(df[['ecg_id', 'filename_hr']].to_numpy(), df['label'])
    df_balanced = pd.DataFrame(X_res, columns=['ecg_id', 'filename_hr'])
    df_balanced['label'] = y_res

    return df_balanced


# In[10]:


# Collecting and labeling dataset
df_labeled = collect_and_label('ptbxl_database.csv')

# Balancing and augmenting the dataset
df_balanced_and_augmented = balance_and_augment(data)

# Printing the first few rows of the resulting DataFrame
print(df_balanced_and_augmented.head())


# In[11]:


label_counts = Counter(df_balanced_and_augmented['label'])
for label, count in label_counts.items():
    print(f"Label '{label}' has {count} records.")


# In[12]:


df_balanced_and_augmented=df_balanced_and_augmented.sample(frac = 1 , ignore_index=True, random_state=123)


# In[13]:


df_balanced_and_augmented


# In[14]:


# Identify missing values
missing_values = df_balanced_and_augmented.isnull().sum()
print("Missing values:")
print(missing_values)  


# In[15]:


# Display the first record from the DataFrame
print(df_balanced_and_augmented.iloc[200])


# In[16]:


from scipy.signal import butter, filtfilt

# Apply Butterworth high-pass filter
def apply_highpass_filter(signal, lowcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='high')
    return filtfilt(b, a, signal)

# Function to apply high-pass filter to lead 1 at channel 0
def apply_highpass_filter_to_lead1(file_path):
    # Load ECG record
    record = wfdb.rdrecord(file_path)

    # Extract lead 1 signal (assuming it's at channel 0)
    lead_1_signal = record.p_signal[:, 0]

    # Sampling frequency
    fs = record.fs

    # Apply high-pass filter
    lowcut = 0.5  # Adjust cutoff frequency as needed
    filtered_lead_1_signal = apply_highpass_filter(lead_1_signal, lowcut, fs)

    return filtered_lead_1_signal

# Assuming 'filename_hr' contains file paths to ECG signals
file_paths = df_balanced_and_augmented['filename_hr']

# Apply high-pass filter to lead 1 at channel 0 for each file
filtered_signals = []
for file_path in file_paths:
    filtered_signal = apply_highpass_filter_to_lead1(file_path)
    filtered_signals.append(filtered_signal)

# Add filtered signals to DataFrame
df_balanced_and_augmented['filtered_lead_1_channel_0'] = filtered_signals

# Display DataFrame with filtered signals
print(df_balanced_and_augmented)


# In[17]:


import matplotlib.pyplot as plt
from tensorflow import data 
import wfdb as sig
import keras
from keras import regularizers
from sklearn.model_selection import train_test_split


# In[18]:


def preprocess(dat):
    data_dir = list(dat['filename_hr'])
    data_signal = map(read_signal , data_dir)
    data_signal = list(data_signal)
    data_signal = np.array(data_signal)
    data_dict = {'NORM' : 0 , 'IMI': 1  }
    encoded_label = dat['label'].map(data_dict)
    return np.array(data_signal)  , np.array(encoded_label)


# In[19]:


from sklearn.model_selection import train_test_split

# Splitting data into train and (validation + test) sets
train, val_test = train_test_split(df_balanced_and_augmented, train_size=0.7, random_state=1002)

# Splitting (validation + test) into validation and test sets
validation, test = train_test_split(val_test, test_size=0.5, random_state=1002)

# Printing the sizes of train, validation, and test sets
print("Train size:", len(train))
print("Validation size:", len(validation))
print("Test size:", len(test))


# In[20]:


print(train.shape)
print(validation.shape)
print(test.shape)


# In[21]:


def read_signal(record):
    tes = sig.rdrecord(record,sampfrom=0 , sampto=5000)
    signal = tes.__dict__['p_signal'][::,0]
    return signal


# In[22]:


X_train , y_train = preprocess(train)
X_valid , y_valid = preprocess(validation)
X_test  , y_test  = preprocess(test) 


# In[23]:


print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(X_test.shape)
print(y_test.shape)


# In[24]:


# Reshape input data to add the timestep dimension
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[25]:


print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(X_test.shape)
print(y_test.shape)


# In[30]:


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

# Define input shape and number of classes
input_shape = X_train.shape[1:]
num_classes = 2  # Two classes: normal and abnormal

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

history = crnn_model.fit(X_train, y_train, 
                         batch_size=batch_size, 
                         epochs=epochs, 
                         validation_data=(X_valid, y_valid),
                         callbacks=[checkpoint])

# Evaluate the model on the test set
test_loss, test_accuracy = crnn_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# In[31]:


# Predict probabilities on the test set
y_pred_proba = crnn_model.predict(X_test)


# In[ ]:


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


# In[32]:


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


# In[44]:


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

