import numpy as np
import pandas as pd
import gensim
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import stopwords
from tqdm import tqdm
from processing_data_and_nn import *

from torch.utils.data import Dataset

# Load the csv file 
df_text = pd.read_csv("preprocessed_bias_train.csv")
# This is the preprocessed text
corpus = df_text['processed_text']
# Should contain tokenized text
tokens = df_text["lemmatized_tokens"]

#Get stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Prepare the Word2Vec model
corpus_tokens = []
for text in tokens:
    corpus_tokens.append(text.split())  # Tokenized form of text

# Build Word2Vec model
model = gensim.models.Word2Vec(
    corpus_tokens,  # tokenized sentences
    vector_size=20,  # size of the embedding vectors
    window=5,  # context window size relative to current main word
    min_count=2,  # ignore words that appear less than 2 times
    workers=12,  
    epochs=10  
)

# takes the text and assigns them to their respective generated value
text_corpus_values = Processing_Text.text_to_value(corpus_tokens, model)

# gets respective bias label values from file (0- unbiased, 1- biased)
bias_labels = df_text['label'] 

# Reset index to ensure that indices are aligned correctly
X_train, X_test, y_train, y_test = train_test_split(
                                                    text_corpus_values, 
                                                    bias_labels, 
                                                    test_size=0.4, 
                                                    random_state=2
)

# Ensure the indices are aligned after splitting
# Also making sure the datasets are in correct format before learning
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train).reset_index(drop=True)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test).reset_index(drop=True)

# Initialize the datasets and dataloaders
# Turning the test/train sets into datasets for network to train on 
batch_size = 32
trainset = Custom_Dataset(X_train, y_train)
testset = Custom_Dataset(X_test, y_test)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Build the neural network model
bd_model = Bias_Detect_Model()

# Loss function + optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer_bdm = optim.Adam(bd_model.parameters(), lr=learning_rate)

# Get the training data converted from previous steps
# Training should occur here
num_epochs_bdm = 15
for epoch in range(num_epochs_bdm):  
    running_loss_bdm = 0.0
   
    for i, data in enumerate(trainloader, 0):
        # get inputs;--- data is a list of [inputs, labels]
        data_inputs, labels = data
        # zero parameter gradients
        optimizer_bdm.zero_grad()
        
        # forward pass
        outputs = bd_model(data_inputs)
        
        # Calculate loss
        # Going backwards + optimizing
        loss = criterion(outputs, labels)
        loss.backward()  # backpropagation
        optimizer_bdm.step()
        
        running_loss_bdm += loss.item()
    
    print("Epoch Num:", epoch)
    print("Training loss:", running_loss_bdm)

print('Finished Training!')

# Save the trained model
torch.save(bd_model.state_dict(), 'bdm.pth')

# Calculating the needed metric for evaluation (Accuracy, Precision, Recall, F1 Score)
bd_model.load_state_dict(torch.load('bdm.pth'))

correct_bdm = 0
total_bdm = 0

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0 

with torch.no_grad():
    for data in testloader:
        text, labels = data
        outputs = bd_model(text)
        _, predicted = torch.max(outputs, 1)
        
        correct_bdm += (predicted == labels).sum().item()
        total_bdm += labels.size(0)
        
        # calculate the true pos/neg and false pos/neg -- metrics
        true_pos += ((predicted == 1) & (labels == 1)).sum().item()
        false_pos += ((predicted == 1) & (labels == 0)).sum().item()
        false_neg += ((predicted == 0) & (labels == 1)).sum().item()
        true_neg += ((predicted == 0) & (labels == 0)).sum().item()

accuracy = correct_bdm / total_bdm
print('Accuracy for network: ', accuracy)

precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
print('Precision for network: ', precision)

recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
print('Recall for network: ', recall)

f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
print('F1 for network: ', f1)

