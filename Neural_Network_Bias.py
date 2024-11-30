import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import ast  # For safely evaluating strings as Python objects

# Load the preprocessed data
bias_train = pd.read_csv('TFIDF/tfidf_bias_train_custom.csv')
bias_valid = pd.read_csv('TFIDF/tfidf_bias_valid_custom.csv')

print(bias_train.columns)
print(bias_valid.columns)
print(bias_train['label'].value_counts())

# Load and clean LDA results
lda_results = pd.read_csv('Custom LDA Topics/doc_topic_mappings.csv')

# Function to clean and parse the 'Topic Distribution' column
def clean_topic_distribution(dist_str):
    try:
        dist_str = dist_str.strip().replace("'", "")  # Remove stray characters
        return ast.literal_eval(dist_str)  # Safely parse as Python list
    except (ValueError, SyntaxError):
        print(f"Error parsing: {dist_str}")
        return None  # Handle invalid rows gracefully

# Apply cleaning function to the LDA results
lda_results['Topic Distribution'] = lda_results['Topic Distribution'].apply(clean_topic_distribution)

# Drop rows with invalid 'Topic Distribution'
lda_results = lda_results.dropna(subset=['Topic Distribution'])

# Function to prepare input data
def prepare_input_data(tfidf_df, lda_df, word_list=None):
    # Parse TF-IDF features from strings to dictionaries
    tfidf_features = [eval(features) for features in tfidf_df['tfidf_features']]
    
    all_words = set()
    for doc in tfidf_features:
        all_words.update(doc.keys())
    
    # If no word list is provided, create one based on the documents
    if word_list is None:
        word_list = sorted(list(all_words))
    
    tfidf_matrix = []
    for doc in tfidf_features:
        doc_vector = [doc.get(word, 0) for word in word_list]
        tfidf_matrix.append(doc_vector)
    
    tfidf_features = np.array(tfidf_matrix)
    
    # Convert the LDA 'Topic Distribution' column into a NumPy array
    lda_features = np.array(lda_df['Topic Distribution'].tolist())
    
    if lda_features.ndim == 1:
        lda_features = lda_features.reshape(-1, 1)
    
    # Align TF-IDF and LDA data sizes
    min_samples = min(tfidf_features.shape[0], lda_features.shape[0])
    tfidf_features = tfidf_features[:min_samples]
    lda_features = lda_features[:min_samples]
    
    # Combine TF-IDF and LDA features
    return np.hstack((tfidf_features, lda_features)), word_list

# Prepare training data and obtain word list
X_train, word_list = prepare_input_data(bias_train, lda_results)

# Prepare validation data using the same word list
X_valid, _ = prepare_input_data(bias_valid, lda_results.iloc[:len(bias_valid)], word_list)

# Check the shapes of the data
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_valid.shape}")

# Prepare labels
y_train = bias_train['label'].iloc[:X_train.shape[0]]
y_valid = bias_valid['label'].iloc[:X_valid.shape[0]]

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_valid_encoded = le.transform(y_valid)

# Convert labels to categorical format
y_train_cat = to_categorical(y_train_encoded)
y_valid_cat = to_categorical(y_valid_encoded)

# Define model architecture
input_dim = X_train.shape[1]
num_classes = len(le.classes_)

print(f"Input dimension (features): {input_dim}")

model = Sequential([
    Dense(256, activation='relu', input_dim=input_dim),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_valid, y_valid_cat),
                    epochs=3,
                    batch_size=32,
                    verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_valid, y_valid_cat, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Prediction function
def predict_bias(new_text, model, word_list, lda_results):
    # Create TF-IDF vector for the new text
    tfidf_features = {word: 0 for word in word_list}
    for word in new_text.split():
        if word in tfidf_features:
            tfidf_features[word] += 1

    tfidf_vector = [tfidf_features[word] for word in word_list]

    # Simulate LDA topic distribution for the new text
    lda_array = np.array(lda_results['Topic Distribution'].tolist())
    lda_vector = lda_array.mean(axis=0)  # Compute the mean topic distribution
    lda_vector = lda_vector.reshape(1, -1)

    # Combine TF-IDF and LDA vectors
    combined_features = np.hstack((np.array(tfidf_vector).reshape(1, -1), lda_vector))

    # Make prediction
    prediction = model.predict(combined_features)
    predicted_class = le.inverse_transform(np.argmax(prediction, axis=1))[0]

    return predicted_class

# Example usage
new_article = "Coach Smith is eager to snatch up Carter from the rookie pool. No one else sees his potential, but Smith is convinced he's found the next Patrick Mahomes. Carter's ability to read defenses is on another level for a rookie. Even Aaron Rodgers wasn't this sharp in his first year."
predicted_bias = predict_bias(new_article, model, word_list, lda_results)
print(f"Predicted bias: {predicted_bias}")


# Get predictions on the validation set
y_pred = model.predict(X_valid)
y_pred_classes = np.argmax(y_pred, axis=1)  # Predicted class labels
y_true_classes = y_valid_encoded  # True class labels

# Classification metrics
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=[str(cls) for cls in le.classes_]))

# Confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)