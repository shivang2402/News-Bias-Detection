import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from src.custom_lda_v2 import CustomLDA  # Replace with the actual file name where CustomLDA is defined
import pickle

# Create Insights folder if it does not exist
output_folder = "Insights"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the trained LDA model from the file
with open('lda_model.pkl', 'rb') as f:
    lda = pickle.load(f)

# Load the saved results
# Document-topic mappings and topic-aspect mappings from previous computations
doc_topic_df = pd.read_csv('results/Custom LDA Topics/doc_topic_mappings.csv')
topic_aspect_df = pd.read_csv('results/Custom LDA Topics/doc_topic_mappings.csv')

# 1. Visualize Topic Distribution
plt.figure(figsize=(8, 5))
doc_topic_df['Dominant Topic'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Topic Distribution')
plt.xlabel('Topic ID')
plt.ylabel('Number of Documents')
plt.savefig(f"{output_folder}/topic_distribution.png")
plt.close()

# 2. Generate Word Clouds for Topics
for topic_id in topic_aspect_df['Topic ID']:
    # Extract keywords for the current topic
    keywords = topic_aspect_df.loc[topic_aspect_df['Topic ID'] == topic_id, 'Keywords'].values[0]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keywords)

    # Display and save the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axes for cleaner visualization
    plt.title(f'Topic {topic_id} Word Cloud')
    plt.savefig(f"{output_folder}/topic_{topic_id}_wordcloud.png")
    plt.close()

# 3. Stacked Bar Chart for Label-Topic Distribution
label_topic_dist = doc_topic_df.groupby(['Dominant Topic', 'Label']).size().unstack(fill_value=0)
label_topic_dist.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Label Distribution Across Topics')
plt.xlabel('Topic ID')
plt.ylabel('Number of Documents')
plt.savefig(f"{output_folder}/label_topic_distribution.png")
plt.close()

# Advanced Metrics and Visualizations

# 1. Compute Inter-topic Similarity
def compute_inter_topic_similarity(lda):
    similarity_matrix = cosine_similarity(lda.topic_word_counts)
    return similarity_matrix

# Visualize inter-topic similarity as a heatmap
similarity_matrix = compute_inter_topic_similarity(lda)
sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", 
            xticklabels=range(lda.num_topics), yticklabels=range(lda.num_topics))
plt.title('Inter-topic Similarity Heatmap')
plt.savefig(f"{output_folder}/inter_topic_similarity.png")
plt.close()

# 2. Perplexity Calculation
def compute_perplexity(lda):
    likelihood = 0
    for doc_idx, doc in enumerate(lda.corpus):
        for word, tfidf_score in doc.items():
            word_id = lda.word_to_id[word]
            topic_probs = (lda.doc_topic_counts[doc_idx] + lda.alpha) * \
                          (lda.topic_word_counts[:, word_id] + lda.beta) / \
                          (lda.topic_counts + lda.beta * lda.vocab_size)
            likelihood += np.log(np.sum(topic_probs))
    perplexity = np.exp(-likelihood / sum(len(doc) for doc in lda.corpus))
    return perplexity

# 3. Topical Diversity
def compute_topical_diversity(lda, top_n=10):
    unique_words = set()
    total_words = 0
    for topic_id in range(lda.num_topics):
        word_probs = lda.topic_word_counts[topic_id] / lda.topic_counts[topic_id]
        top_word_ids = np.argsort(word_probs)[::-1][:top_n]
        unique_words.update(top_word_ids)
        total_words += len(top_word_ids)
    topical_diversity = len(unique_words) / total_words
    return topical_diversity

# Calculate and log perplexity and topical diversity
perplexity = compute_perplexity(lda)
diversity = compute_topical_diversity(lda)
logging.info(f"Perplexity: {perplexity}")
logging.info(f"Topical Diversity: {diversity}")

# 4. Label Purity
def compute_label_purity(lda, labels):
    topic_purity = []
    for topic_id in range(lda.num_topics):
        label_counts = {0: 0, 1: 0}
        for doc_idx, label in enumerate(labels):
            if np.argmax(lda.doc_topic_counts[doc_idx]) == topic_id:
                label_counts[label] += 1
        purity = max(label_counts.values()) / sum(label_counts.values()) if sum(label_counts.values()) > 0 else 0
        topic_purity.append(purity)
    return topic_purity

label_purity = compute_label_purity(lda, doc_topic_df['Label'])
logging.info(f"Label Purity: {label_purity}")

# 5. Topic Drift Over Iterations
def compute_topic_drift(lda):
    return np.var(lda.doc_topic_counts, axis=0)

topic_drift = compute_topic_drift(lda)
plt.plot(range(len(topic_drift)), topic_drift, color='purple')
plt.title('Topic Drift Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Variance in Topic Distributions')
plt.savefig(f"{output_folder}/topic_drift.png")
plt.close()

# 6. Matrix Sparsity
def compute_sparsity(matrix):
    total_elements = np.prod(matrix.shape)
    non_zero_elements = np.count_nonzero(matrix)
    return 1 - (non_zero_elements / total_elements)

sparsity = compute_sparsity(lda.doc_topic_counts)
logging.info(f"Sparsity: {sparsity}")

# 7. Model Convergence
def plot_model_convergence(lda):
    changes = np.diff(lda.topic_word_counts, axis=0)
    avg_change = np.mean(np.abs(changes), axis=1)
    plt.plot(range(len(avg_change)), avg_change, color='orange')
    plt.title('Model Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Avg Change in Topic-Word Distributions')
    plt.savefig(f"{output_folder}/model_convergence.png")
    plt.close()

plot_model_convergence(lda)

# Save matrices to text files
np.savetxt(f"{output_folder}/doc_topic_counts.txt", lda.doc_topic_counts, fmt='%f')
np.savetxt(f"{output_folder}/topic_word_counts.txt", lda.topic_word_counts, fmt='%f')
np.savetxt(f"{output_folder}/similarity_matrix.txt", similarity_matrix, fmt='%f')

# Save the DataFrame as a CSV file
doc_topic_df.to_csv(f"{output_folder}/doc_topic_mappings.csv", index=False)
topic_aspect_df.to_csv(f"{output_folder}/lda_topics_with_aspects.csv", index=False)
