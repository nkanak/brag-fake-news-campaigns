import json
import os
from collections import defaultdict
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import networkx as nx
import statistics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_result_file(num_shots):
    return f'result_rag_{2*num_shots}_shots_contrastive_with_graph.json'

def extract_text_and_metrics(data):
    text = data['nodes'][0]['tweet_text']
    G = nx.Graph()
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'])
    
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    retweet_times = [node['delay'] for node in data['nodes'][1:]]
    median_retweet_time = statistics.median(retweet_times) if retweet_times else 0
    
    metrics = {
        'avg_degree_centrality': sum(degree_centrality.values()) / len(degree_centrality),
        'eigenvector_centrality': sum(eigenvector_centrality.values()) / len(eigenvector_centrality),
        'median_retweet_time': median_retweet_time
    }
    
    return text, metrics

def load_results(result_file):
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return {}

def save_result(results, filename, prediction, result_file):
    results[filename] = prediction
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_and_process_data(folder_path, vectorizer=None, fit=False):
    cache_file = os.path.join(CACHE_DIR, f'{os.path.basename(folder_path)}_vectors.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    data = []
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            text, metrics = extract_text_and_metrics(json_data)
            data.append({
                'filename': filename,
                'text': text,
                'metrics': metrics,
                'label': json_data['label']
            })
            texts.append(text)

    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(texts)
    elif fit:
        vectors = vectorizer.fit_transform(texts)
    else:
        vectors = vectorizer.transform(texts)
    
    result = (data, vectors, vectorizer)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result

def find_most_similar_samples(test_vector, train_vectors, num_shots):
    similarities = cosine_similarity(test_vector, train_vectors)[0]
    return np.argsort(similarities)[-num_shots:][::-1]

def add_contrastive_samples(selected_indices, train_data, train_vectors):
    contrastive_indices = []
    for idx in selected_indices:
        sample_vector = train_vectors[idx]
        similarities = cosine_similarity(sample_vector, train_vectors)[0]
        
        for contrast_idx in np.argsort(similarities)[::-1]:
            if train_data[contrast_idx]['label'] != train_data[idx]['label']:
                contrastive_indices.append(contrast_idx)
                break
    
    return np.concatenate([selected_indices, contrastive_indices])

def generate_prompt_template(num_shots):
    prompt = '''You are an intelligent classifier capable of identifying political fake news campaigns. Your task is to analyze the propagation of a given tweet, focusing on both the text content and aggregated network metrics to detect any coordinated behavior that might indicate a political fake news campaign. You will be provided with the tweet text and a set of statistics derived from its retweet propagation graph.

    The following centrality metrics are important for your analysis:

    - Degree centrality measures how many direct connections a user (node) has. A high degree centrality indicates that the user is directly connected to many others, suggesting they play a central role in propagating the tweet.
    - Eigenvector centrality measures the influence of a user within the network. It considers not only the number of connections a user has, but also the importance of those connections.
    - Median retweet time refers to how quickly the tweet is typically retweeted. In coordinated campaigns, retweets often occur in rapid bursts, indicating synchronized activity.

    Base your decision on the text content, language use, messaging tone, and these network metrics.

    Here are some labeled examples of tweets to help guide your analysis. The labels indicate whether the tweet was part of a "Fake" coordinated political campaign or "Real" organic behavior. Pay attention to the differences between Fake and Real examples.

    '''
    for i in range(1, num_shots * 2 + 1):
        prompt += f'''
    Example {i} ({'Similar' if i <= num_shots else 'Contrastive'}):
    Tweet text: "{{example_{i}}}"
    Average degree centrality: {{avg_degree_centrality_{i}:.2f}}
    Eigenvector centrality: {{eigenvector_centrality_{i}:.2f}}
    Median retweet time: {{median_retweet_time_{i}:.2f}} seconds
    Label: {{label_{i}}}
    '''
    
    prompt += '''
    Now, analyze the following input tweet and determine whether it suggests a "Fake" coordinated effort or "Real" organic behavior.'''
    
    return prompt

def check_astroturfing(text, metrics, all_examples, prompt_template):
    example_dict = {}
    for i, example in enumerate(all_examples, 1):
        example_dict.update({
            f'example_{i}': example['text'],
            f'avg_degree_centrality_{i}': example['metrics']['avg_degree_centrality'],
            f'eigenvector_centrality_{i}': example['metrics']['eigenvector_centrality'],
            f'median_retweet_time_{i}': example['metrics']['median_retweet_time'],
            f'label_{i}': example['label'].capitalize()
        })
    
    prompt = prompt_template.format(**example_dict)
    
    response = ollama.chat(model='llama3.1:70b', messages=[
        {
            'role': 'system',
            'content': prompt
        },
        {
            'role': 'user',
            'content': f'''Tweet text: {text}
            Average degree centrality: {metrics['avg_degree_centrality']:.2f}
            Eigenvector centrality: {metrics['eigenvector_centrality']:.2f}
            Median retweet time: {metrics['median_retweet_time']:.2f} seconds

            Output:
            Fake or Real (No explanation needed).'''
        },
    ],
    options={
        "temperature": 0
    })
    return response['message']['content'].strip().lower()

def process_files(test_folder, train_folder, num_shots):
    result_file = get_result_file(num_shots)
    results = load_results(result_file)
    
    # Load and process train data
    train_data, train_vectors, vectorizer = load_and_process_data(train_folder, fit=True)
    
    # Load and process test data using the same vectorizer
    test_data, test_vectors, _ = load_and_process_data(test_folder, vectorizer=vectorizer)
    
    prompt_template = generate_prompt_template(num_shots)
    
    with tqdm(total=len(test_data), desc="Processing files") as pbar:
        for i, test_item in enumerate(test_data):
            if test_item['filename'] in results:
                pbar.update(1)
                continue
            
            text = test_item['text']
            metrics = test_item['metrics']
            test_vector = test_vectors[i]
            
            # Find most similar examples from train data
            similar_indices = find_most_similar_samples(test_vector, train_vectors, num_shots)
            
            # Add contrastive samples
            all_indices = add_contrastive_samples(similar_indices, train_data, train_vectors)
            
            all_examples = [train_data[idx] for idx in all_indices]
            
            prediction = check_astroturfing(text, metrics, all_examples, prompt_template)
            
            save_result(results, test_item['filename'], prediction, result_file)
            
            pbar.update(1)
    
    return results

def calculate_metrics(test_data, results):
    true_labels = []
    predicted_labels = []
    
    for item in test_data:
        filename = item['filename']
        if filename in results:
            true_labels.append(1 if item['label'] == 'fake' else 0)
            predicted_labels.append(1 if results[filename] == 'fake' else 0)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    return accuracy, precision, recall, f1

def main(num_shots):
    test_folder = './dataset1/test'  # Adjust this to your test folder path
    train_folder = './dataset1/train'  # Adjust this to your train folder path
    
    results = process_files(test_folder, train_folder, num_shots)
    
    # Load test data for metric calculation
    test_data, _, _ = load_and_process_data(test_folder)
    
    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(test_data, results)
    
    # Print metrics
    print(f"Results for {2*num_shots}-shot RAG with similarity-based selection, contrastive samples, and graph metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    num_shots = 3
    main(num_shots)