import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV files
doc_topic_mappings = pd.read_csv('Custom LDA Topics/doc_topic_mappings.csv')  # Contains document IDs and topics
lda_topics_with_aspects = pd.read_csv('Custom LDA Topics/lda_topics_with_aspects.csv')  # Topics with human-readable aspects
tfidf_bias_train = pd.read_csv('TFIDF/tfidf_bias_train_custom.csv')  # TF-IDF features for documents
tfidf_aspects = pd.read_csv('TFIDF/tfidf_aspects_custom.csv')  # TF-IDF features for aspects

# Step 2: Convert TF-IDF features from strings to usable formats
# Convert TF-IDF bias train features from strings to dictionaries
tfidf_bias_train['tfidf_features'] = tfidf_bias_train['tfidf_features'].apply(ast.literal_eval)
# Convert the dictionaries into a DataFrame
tfidf_features = tfidf_bias_train['tfidf_features'].apply(pd.Series)

# Combine the TF-IDF features with the remaining columns, excluding 'text'
tfidf_bias_train = pd.concat([tfidf_features, tfidf_bias_train.drop(columns=['text', 'tfidf_features'])], axis=1)

# Convert TF-IDF aspects from strings to dictionaries
tfidf_aspects['tfidf_features'] = tfidf_aspects['tfidf_features'].apply(ast.literal_eval)
# Convert the dictionaries into a DataFrame
aspect_features = tfidf_aspects['tfidf_features'].apply(pd.Series)

# Combine the document-based features and aspect-based features
X_tfidf = pd.concat([tfidf_bias_train, aspect_features], axis=1)

# Step 3: Normalize the TF-IDF features
scaler = StandardScaler()
X_tfidf_scaled = scaler.fit_transform(X_tfidf)
# Convert back to DataFrame for easier manipulation
X_tfidf_scaled = pd.DataFrame(X_tfidf_scaled, columns=X_tfidf.columns)

# Step 4: Prepare the LDA Topic Distribution features
# Convert 'Topic Distribution' from string to list
doc_topic_mappings['Topic Distribution'] = doc_topic_mappings['Topic Distribution'].apply(ast.literal_eval)

# Create a DataFrame with topic distribution as separate columns
topic_columns = pd.DataFrame(doc_topic_mappings['Topic Distribution'].to_list(),
                             columns=[f'Topic_{i}' for i in range(len(doc_topic_mappings['Topic Distribution'][0]))])

# Combine TF-IDF features with topic distribution features
X_final = pd.concat([X_tfidf_scaled, topic_columns], axis=1)

# Step 5: Prepare the target labels
y_final = doc_topic_mappings['Label']  # Assuming 'Label' is your target

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Step 7: Define the neural network model
model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input layer
    layers.Dense(128, activation='relu'),  # Hidden layer with ReLU activation
    layers.Dense(64, activation='relu'),   # Another hidden layer
    layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Train the neural network
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 9: Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Predicting classes, thresholding at 0.5

# Compute the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict(X_test))

# Print evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')

# Step 10: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Optional: Save the trained model
model.save('trained_model.h5')
