#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb 
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import StratifiedKFold
#from ecgdetectors import Detectors
from tensorflow import data 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import regularizers
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
file_path = r"ptbxl_database.csv"
data = pd.read_csv(file_path)

# Count the number of rows where 'scp_codes' contains 'NORM'
count_norm = data[data['scp_codes'].str.contains('NORM')].shape[0]

# Count the number of rows where 'scp_codes' contains 'IMI'
count_imi = data[data['scp_codes'].str.contains('IMI')].shape[0]

# Print the results
print("Number of records with 'NORM' in scp_codes column:", count_norm)
print("Number of records with 'IMI' in scp_codes column:", count_imi)



# In[5]:


import pandas as pd

# Load the dataset
data = pd.read_csv(r"ptbxl_database.csv")

# Filter the dataset based on conditions
filtered_data = data[(data['burst_noise'].isnull()) & 
                     (data['electrodes_problems'].isnull()) & 
                     (data['extra_beats'].isnull())]

# Filter for records containing 'NORM' and 'IMI' in 'scp_codes' column
norm_data = filtered_data[filtered_data['scp_codes'].str.contains('NORM')]
imi_data = filtered_data[filtered_data['scp_codes'].str.contains('IMI')]

# Take the same number of records for 'NORM' and 'IMI'
min_records = min(norm_data.shape[0], imi_data.shape[0])
norm_data = norm_data.sample(n=min_records, random_state=42)
imi_data = imi_data.sample(n=min_records, random_state=42)

# Concatenate the two subsets
final_data = pd.concat([norm_data, imi_data])

# Save the concatenated data to a new CSV file
final_data.to_csv('New data.csv', index=False)

# Print confirmation message
print("Dataset saved successfully.")


# In[ ]:




