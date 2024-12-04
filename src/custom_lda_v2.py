import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the preprocessed bias data and aspects data (already saved as CSVs)
logging.info("Loading data...")
bias_data = pd.read_csv('results/TFIDF/tfidf_bias_train_custom.csv')  # CSV with TF-IDF values for the bias data
aspects_data = pd.read_csv('results/TFIDF/tfidf_bias_train_custom.csv')  # CSV with TF-IDF values for the aspects data

# Convert the TF-IDF DataFrame into the required format for the LDA
def convert_tfidf_to_corpus(df):
    logging.info(f"Converting DataFrame to corpus with {len(df)} documents.")
    corpus = []
    labels = []
    for index, row in df.iterrows():
        doc_tfidf = eval(row['tfidf_features'])  # Convert string dict to actual dict
        corpus.append(doc_tfidf)
        if 'label' in row:  # Extract labels if present
            labels.append(row['label'])
    return corpus, labels

bias_corpus, bias_labels = convert_tfidf_to_corpus(bias_data)
aspects_corpus, _ = convert_tfidf_to_corpus(aspects_data)


# Custom LDA class
class CustomLDA:
    def __init__(self, corpus, num_topics=15, alpha=0.01, beta=0.01, num_iterations=1000):
        logging.info("Initializing CustomLDA...")
        self.corpus = corpus
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.num_iterations = num_iterations
        self.vocab = set(word for doc in corpus for word in doc.keys())
        self.vocab_size = len(self.vocab)
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.doc_topic_counts = np.zeros((len(corpus), num_topics))
        self.topic_word_counts = np.zeros((num_topics, self.vocab_size))
        self.topic_counts = np.zeros(num_topics)
        self.word_assignments = []

        for doc_idx, doc in enumerate(corpus):
            word_assignments_doc = []
            for word, tfidf_score in doc.items():
                word_id = self.word_to_id[word]
                topic = np.random.randint(0, num_topics)
                word_assignments_doc.append(topic)
                self.doc_topic_counts[doc_idx, topic] += tfidf_score
                self.topic_word_counts[topic, word_id] += tfidf_score
                self.topic_counts[topic] += tfidf_score
            self.word_assignments.append(word_assignments_doc)
        logging.info(f"CustomLDA initialized with {self.num_topics} topics.")

    def sample_topic(self, doc_idx, word_idx, word_id, tfidf_score):
        current_topic = self.word_assignments[doc_idx][word_idx]
        self.doc_topic_counts[doc_idx, current_topic] -= tfidf_score
        self.topic_word_counts[current_topic, word_id] -= tfidf_score
        self.topic_counts[current_topic] -= tfidf_score
        topic_probs = (self.doc_topic_counts[doc_idx] + self.alpha) * \
                      (self.topic_word_counts[:, word_id] + self.beta) / \
                      (self.topic_counts + self.beta * self.vocab_size)
        topic_probs /= topic_probs.sum()
        new_topic = np.random.choice(self.num_topics, p=topic_probs)
        self.doc_topic_counts[doc_idx, new_topic] += tfidf_score
        self.topic_word_counts[new_topic, word_id] += tfidf_score
        self.topic_counts[new_topic] += tfidf_score
        return new_topic

    def train(self):
        logging.info("Starting training...")
        for iteration in range(self.num_iterations):
            logging.info(f"Iteration {iteration + 1}/{self.num_iterations}")
            for doc_idx, doc in enumerate(self.corpus):
                for word_idx, (word, tfidf_score) in enumerate(doc.items()):
                    word_id = self.word_to_id[word]
                    new_topic = self.sample_topic(doc_idx, word_idx, word_id, tfidf_score)
                    self.word_assignments[doc_idx][word_idx] = new_topic
        logging.info("Training completed.")

    def print_topics(self, top_n=1):
        logging.info("Printing topics...")
        for topic_id in range(self.num_topics):
            word_probs = self.topic_word_counts[topic_id] / self.topic_counts[topic_id]
            top_word_ids = np.argsort(word_probs)[::-1][:top_n]
            top_words = [self.id_to_word[word_id] for word_id in top_word_ids]
            print(f"Topic {topic_id}: {', '.join(top_words)}")
   
# Expand top keywords with related words using cosine similarity of TF-IDF vectors
def expand_keywords_with_similarity(top_keywords, corpus, top_n=15):
    logging.info(f"Expanding keywords: {top_keywords}")
    expanded_keywords = set(top_keywords)
    word_to_index = {word: i for i, word in enumerate(corpus[0].keys())}
    tfidf_matrix = []
    for doc in corpus:
        tfidf_vector = np.zeros(len(word_to_index))
        for word, tfidf_score in doc.items():
            word_index = word_to_index.get(word, None)
            if word_index is not None:
                tfidf_vector[word_index] = tfidf_score
        tfidf_matrix.append(tfidf_vector)
    tfidf_matrix = np.array(tfidf_matrix)
    for keyword in top_keywords:
        keyword_vector = np.zeros(len(word_to_index))
        keyword_index = word_to_index.get(keyword, None)
        if keyword_index is not None:
            keyword_vector[keyword_index] = 1.0
        cosine_similarities = cosine_similarity([keyword_vector], tfidf_matrix)[0]
        similar_words_idx = np.argsort(cosine_similarities)[::-1][:top_n]
        for idx in similar_words_idx:
            related_word = list(corpus[idx].keys())[0]
            expanded_keywords.add(related_word)
    logging.info(f"Expanded keywords: {expanded_keywords}")
    return list(expanded_keywords)

# Map topic to the most similar aspect using cosine similarity and expanded keywords
def map_topics_to_aspects(lda, aspects_data, top_n_words=15):
    logging.info("Mapping topics to aspects...")
    lda_vocab = [lda.id_to_word[i] for i in range(lda.vocab_size)]
    aspect_mappings = []
    for topic_id in range(lda.num_topics):
        word_probs = lda.topic_word_counts[topic_id] / lda.topic_counts[topic_id]
        top_word_ids = np.argsort(word_probs)[::-1][:top_n_words]
        top_words = [lda.id_to_word[word_id] for word_id in top_word_ids]
        expanded_keywords = expand_keywords_with_similarity(top_words, aspects_corpus, top_n=15)
        aspect_tfidf_matrix = []
        word_to_index = {word: i for i, word in enumerate(expanded_keywords)}
        for _, row in aspects_data.iterrows():
            doc_tfidf = eval(row['tfidf_features'])
            tfidf_vector = np.zeros(len(expanded_keywords))
            for word, score in doc_tfidf.items():
                if word in word_to_index:
                    tfidf_vector[word_to_index[word]] = score
            aspect_tfidf_matrix.append(tfidf_vector)
        aspect_tfidf_matrix = np.array(aspect_tfidf_matrix)
        topic_vector = np.zeros(len(expanded_keywords))
        for word_id, prob in zip(top_word_ids, word_probs[top_word_ids]):
            word = lda.id_to_word[word_id]
            if word in word_to_index:
                topic_vector[word_to_index[word]] = prob
        similarity_scores = cosine_similarity([topic_vector], aspect_tfidf_matrix)
        most_similar_aspect_idx = similarity_scores.argmax()
        most_similar_aspect = aspects_data.iloc[most_similar_aspect_idx]['Aspect']
        aspect_mappings.append((topic_id, most_similar_aspect, expanded_keywords))
        logging.info(f"Topic {topic_id} mapped to aspect: {most_similar_aspect} with keywords {expanded_keywords}")
    return aspect_mappings

# Train the LDA model
lda = CustomLDA(bias_corpus, num_topics=15, num_iterations=1000)
lda.train()

import pickle
# Save the trained model to a file
with open('lda_model.pkl', 'wb') as f:
    pickle.dump(lda, f)

# Map topics to aspects and include labels
logging.info("Mapping topics to aspects and including labels...")
topic_aspect_mappings = []
for doc_idx, doc_label in enumerate(bias_labels):  # Use labels from bias data
    doc_topics = lda.doc_topic_counts[doc_idx]
    dominant_topic = np.argmax(doc_topics)  # Get dominant topic for the document
    topic_aspect_mappings.append({
        'Document ID': doc_idx,
        'Label': doc_label,
        'Dominant Topic': dominant_topic,
        'Topic Distribution': lda.doc_topic_counts[doc_idx].tolist()
    })

# Map topics to aspects (if relevant) and save the results
topic_aspect_results = []
aspect_mappings = map_topics_to_aspects(lda, aspects_data)
for topic_id, aspect, keywords in aspect_mappings:
    topic_aspect_results.append({
        'Topic ID': topic_id,
        'Aspect': aspect,
        'Keywords': ', '.join(keywords)
    })

# Save document-level mappings with labels
logging.info("Saving document-level topic-label mappings...")
doc_topic_df = pd.DataFrame(topic_aspect_mappings)
doc_topic_output_file = 'Custom LDA Topics/doc_topic_mappings.csv'
doc_topic_df.to_csv(doc_topic_output_file, index=False)

# Save topic-aspect mappings
logging.info("Saving topic-aspect mappings...")
topic_aspect_df = pd.DataFrame(topic_aspect_results)
topic_aspect_output_file = 'Custom LDA Topics/lda_topics_with_aspects.csv'
topic_aspect_df.to_csv(topic_aspect_output_file, index=False)

logging.info(f"Document-level mappings saved to {doc_topic_output_file}")
logging.info(f"Topic-aspect mappings saved to {topic_aspect_output_file}")


