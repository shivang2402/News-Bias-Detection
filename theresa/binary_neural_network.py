import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gensim 
import keras
import nltk
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
nltk.download('stopwords')
from keras import layers
from theresa.processing_data_and_nn import *

# Load the preprocessed file
df_text = pd.read_csv("../News-Bias-Detection/preprocessed_bias_train.csv")
corpus = df_text['lemmatized_tokens']  # Assuming this is already tokenized

# Directly use the preprocessed corpus
tokens = corpus  # If already tokenized

# Build Word2Vec model
model = gensim.models.Word2Vec(
    tokens,  # Tokenized corpus
    vector_size=150,
    window=5,
    min_count=2,
    workers=10,
    epochs=10
)

# Convert text to embedding values
text_corpus_values = []
for text_token in tokens:
    text_corpus_values.append(Processing_Text.text_to_value(text_token, model))

# Continue with the rest of the process
bias_labels = df_text['label']
text_values = np.array(text_corpus_values)
X_train, X_test, y_train, y_test = train_test_split(text_values, bias_labels, test_size=0.2, random_state=4)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# Prepare datasets and loaders
trainset = Custom_Dataset(X_train, y_train)
testset = Custom_Dataset(X_test, y_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=80, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=80, shuffle=False)

# Build, train, and evaluate the model (same as in your original code)


# Build the neural network model
bd_model = Bias_Detect_Model()

# loss function + optimizer
learning_rate = .01
criterion = nn.CrossEntropyLoss()
optimizer_bdm = torch.optim.Adam(bd_model.parameters(), lr=learning_rate)

# get the training data converted from previous steps
# training should occur here
num_epochs_ffn = 10

for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
    running_loss_ffn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Flatten inputs for ffn
        inputs = torch.flatten(inputs, 1)

        # zero the parameter gradients
        optimizer_bdm.zero_grad()

        # forward + backward + optimize
        outputs = bd_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_bdm.step()
        running_loss_ffn += loss.item()

    print(f"Training loss: {running_loss_ffn}")

print('Finished Training')

torch.save(bd_model.state_dict(), 'bdm.pth')  # saves model file

## calculate the needed metrics
bd_model.load_state_dict(torch.load('bdm.pth'))
correct_bdm = 0
total_bdm = 0

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

# since we're training, we need to calculate the gradients for our outputs
for data in testloader:
    text, labels = data
    outputs_bdm = bd_model(text) 
    _, predicted = torch.max(outputs_bdm, 1)
    correct_bdm += (predicted == labels).sum().item()  # sum for batch accuracy
    total_bdm += labels.size(0)  # total number of labels

    # calculate the true pos/neg and false pos/neg
    true_pos += ((predicted == 1) & (labels == 1)).sum().item()  # element-wise comparison and sum
    false_pos += ((predicted == 1) & (labels == 0)).sum().item()
    false_neg += ((predicted == 0) & (labels == 1)).sum().item()
    true_neg += ((predicted == 0) & (labels == 0)).sum().item()

print('Accuracy for network: ', correct_bdm / total_bdm)

precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
print('Precision for network: ', precision)

recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
print('Recall for network: ', recall)

f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
print('F1 for network: ', f1)
