import io
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk.corpus import stopwords 
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from keras import layers

class Processing_Text():

    #handles the preprocessing of text
    # stop words -  “the,” “a,” “an,” or “in” ...
    def preprocess_text(text, stop_words):
        # lowercase the text
        text = text.lower()
        # remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # tokenize + remove stopwords
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return " ".join(filtered_tokens)

    # takes the text and assigns them to their respective generated value
    # output a value to represent an entire sentence (text_token)
    def text_to_value(texts, model):
        embed_values = []

        for text in texts:
        # Check if the word exists in the model's vocabulary
            if text in model.wv:  # Access the KeyedVectors via `model.wv`
                embed_values.append(model.wv[text])  # Fetch the embedding vector
            else:
                embed_values.append(np.zeros(model.vector_size))  # Use zeros for unknown words

    # Return the mean vector for the sentence
        return np.mean(embed_values, axis=0)


# allow for the network to use it
class Custom_Dataset(Dataset):
    # turns the splits training data into tensor datasets 
    def __init__(self, texts, labels):
        #converts the data into tensor --> 
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):   
        text = torch.tensor(self.texts[idx], dtype=torch.float32)  # Ensure dtype matches
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # CrossEntropyLoss expects long for labels
        return text, label



#  the general model here 
class Bias_Detect_Model(nn.Module):
    def __init__(self, inner_feature=150, outer_feature=10):  # Change inner_feature to 150
        super().__init__()
        self.fc1 = nn.Linear(inner_feature, 588)  # Update fc1's input size
        self.fc2 = nn.Linear(588, 392)
        self.fc3 = nn.Linear(392, 196)
        self.fc4 = nn.Linear(196, 88)
        self.out = nn.Linear(88, outer_feature)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.out(x)
        return x

        

# calculate the needed metrics 
class Metrics():

    def accuracy():
        pass

    def f1():
        pass

    def recall():
        pass

