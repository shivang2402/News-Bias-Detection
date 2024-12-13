import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import ast  # To safely evaluate string representations of Python objects

# Load the data
bias_train = pd.read_csv('results/TFIDF/tfidf_bias_train_custom.csv')
bias_valid = pd.read_csv('results/TFIDF/tfidf_bias_valid_custom.csv')
lda_results = pd.read_csv('results/Custom LDA Topics/doc_topic_mappings.csv')
lda_topics_with_aspects = pd.read_csv('results/Custom LDA Topics/lda_topics_with_aspects.csv')

# Ensure column names are correctly formatted
lda_results.rename(columns=lambda x: x.strip(), inplace=True)
lda_topics_with_aspects.rename(columns=lambda x: x.strip(), inplace=True)

# Debugging
print("LDA Results Columns:", lda_results.columns)
print("LDA Topics Columns:", lda_topics_with_aspects.columns)

# Parse 'Topic Distribution' column
def parse_topic_distribution(distribution):
    try:
        # Convert the string into a list of floats
        return ast.literal_eval(distribution)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing topic distribution: {distribution}. Error: {e}")
        return None

lda_results['Topic Distribution'] = lda_results['Topic Distribution'].apply(parse_topic_distribution)

# Drop rows with invalid topic distributions
lda_results.dropna(subset=['Topic Distribution'], inplace=True)

# Merge LDA results with topics and aspects
merged_lda = lda_results.merge(lda_topics_with_aspects, left_on='Dominant Topic', right_on='Topic ID', how='left')

# Prepare input data
def prepare_input_data(tfidf_df, lda_df, word_list=None):
    # Parse TF-IDF features from training data
    tfidf_features = [eval(features) for features in tfidf_df['tfidf_features']]
    
    all_words = set()
    for doc in tfidf_features:
        all_words.update(doc.keys())
    
    # If no word list is provided, create one based on the documents
    if word_list is None:
        word_list = sorted(list(all_words))
    
    # Create TF-IDF matrix
    tfidf_matrix = []
    for doc in tfidf_features:
        doc_vector = [doc.get(word, 0) for word in word_list]
        tfidf_matrix.append(doc_vector)
    
    tfidf_features = np.array(tfidf_matrix)
    
    # Extract LDA topic distributions
    lda_features = np.array(lda_df['Topic Distribution'].tolist())
    
    # Align the sizes
    min_samples = min(tfidf_features.shape[0], lda_features.shape[0])
    tfidf_features = tfidf_features[:min_samples]
    lda_features = lda_features[:min_samples]
    
    # Combine TF-IDF and LDA features
    return np.hstack((tfidf_features, lda_features)), word_list

# Prepare training and validation datasets
X_train, word_list = prepare_input_data(bias_train, merged_lda)
X_valid, _ = prepare_input_data(bias_valid, merged_lda.iloc[:len(bias_valid)], word_list)

# Check shapes of training and validation data
print(f"Training Data Shape: {X_train.shape}")
print(f"Validation Data Shape: {X_valid.shape}")

# Encode labels
le = LabelEncoder()
y_train = bias_train['label'].iloc[:X_train.shape[0]]
y_valid = bias_valid['label'].iloc[:X_valid.shape[0]]

y_train_encoded = le.fit_transform(y_train)
y_valid_encoded = le.transform(y_valid)

# Convert labels to categorical
y_train_cat = to_categorical(y_train_encoded)
y_valid_cat = to_categorical(y_valid_encoded)

# Define model architecture
input_dim = X_train.shape[1]
num_classes = len(le.classes_)

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
print(f"Test Accuracy: {test_accuracy:.4f}")

# Modified predict function for new text
def predict_bias(new_text, model, word_list, lda_results, lda_topics_with_aspects):
    # Step 1: Generate TF-IDF features for the new text
    tfidf_features = {word: 0 for word in word_list}
    for word in new_text.split():
        if word in tfidf_features:
            tfidf_features[word] += 1
    tfidf_vector = [tfidf_features[word] for word in word_list]
    
    # Step 2: Use the mean of the LDA topic distributions for prediction
    lda_array = np.array(lda_results['Topic Distribution'].tolist())
    lda_vector = lda_array.mean(axis=0)
    
    # Step 3: Merge LDA topics with aspects to get the aspect associated with the dominant topic
    dominant_topic_idx = np.argmax(lda_vector)  # Get the topic with the highest probability
    aspect = lda_topics_with_aspects.loc[lda_topics_with_aspects['Topic ID'] == dominant_topic_idx, 'Aspect'].values[0]
    
    # Step 4: Combine the TF-IDF features and the LDA vector
    combined_features = np.hstack((np.array(tfidf_vector).reshape(1, -1), lda_vector.reshape(1, -1)))
    
    # Step 5: Make the prediction
    prediction = model.predict(combined_features)
    predicted_class = np.argmax(prediction, axis=1)[0]  # 0 for class 0, 1 for class 1
    
    # Step 6: Return the predicted bias (0/1) and the aspect
    return predicted_class, aspect

# Example prediction
# new_article = "President talks about climate change policies and their impact on global markets."
#new_article = "Coach Smith is eager to snatch up Carter from the rookie pool. No one else sees his potential, but Smith is convinced he's found the next Patrick Mahomes. Carter's ability to read defenses is on another level for a rookie. Even Aaron Rodgers wasn't this sharp in his first year."
new_article = "All politicians are corrupt and work only for their own benefit."

predicted_bias, aspect = predict_bias(new_article, model, word_list, merged_lda, lda_topics_with_aspects)

print(f"Predicted Bias: {predicted_bias}")
print(f"Aspect: {aspect}")



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
