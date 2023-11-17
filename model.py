import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os

# load data
current_dir = os.getcwd()
data = pd.read_csv(current_dir + '/dataSets/train-balanced-sarcasm.csv')

# clean data
data = data.dropna()
# Current idea is to see how the model performs with just comment and parent comment
# if we need to in the future I can add the subreddit feature and grab the subreddit description
# for more context 

data = data.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc'], axis=1)

# split data into train and test
train_data, test_data = tfds.load(data, split=['train[:75%]', 'test'])
print(train_data.head())