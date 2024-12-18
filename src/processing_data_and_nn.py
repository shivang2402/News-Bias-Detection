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


    def preprocess_text(text, stop_words):
        '''
        handles the preprocessing of text
        stop words -  “the,” “a,” “an,” or “in” ...
        '''

        # lowercase the text
        text = text.lower()
        # remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # tokenize + remove stopwords
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return " ".join(filtered_tokens)


    def text_to_value(texts, model):
        '''
        takes the text and assigns them to their respective generated value
        output a value to represent an entire sentence (text_token)
        '''
        embed_values = []
        for text in texts:
            word_vectors = []
            for word in text:
                # Check if the word exists in the model's vocabulary
                if word in model.wv:  # access the KeyedVectors via `model.wv`
                    word_vectors.append(model.wv[word])  # Fetch the embedding vector
                else:
                    word_vectors.append(np.zeros(model.vector_size))  # Use zeros for unknown words
            # Return the mean vector for the sentence
            embed_values.append(np.mean(word_vectors, axis=0))

        return np.array(embed_values)


# allow for the network to use it
class Custom_Dataset(Dataset):
    # turns the splits training data into tensor datasets 
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):   
        text = torch.tensor(self.texts.iloc[idx], dtype=torch.float32)  # Ensure dtype matches
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)  # CrossEntropyLoss expects long for labels
        return text, label

#  the general bdm model here 
class Bias_Detect_Model(nn.Module):
    def __init__(self, inner_feature=20, outer_feature=2):
        super(Bias_Detect_Model, self).__init__()
        self.fc1 = nn.Linear(inner_feature, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, outer_feature)  # Binary classification (biased/unbiased)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    