import json
import os
from collections import defaultdict
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_text(data):
    return data['nodes'][0]['tweet_text']

def load_results(result_file):
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return {}

def save_result(results, filename, prediction, result_file):
    results[filename] = prediction
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_and_process_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            text = extract_text(json_data)
            data.append({
                'filename': filename,
                'text': text,
                'label': json_data['label']
            })
    return data

def find_most_similar(test_text, train_data, vectorizer, train_vectors, num_shots):
    test_vector = vectorizer.transform([test_text])
    similarities = cosine_similarity(test_vector, train_vectors)[0]
    top_indices = np.argsort(similarities)[-num_shots:][::-1]
    return [train_data[i] for i in top_indices]

def generate_prompt(similar_examples):
    prompt = '''You are an intelligent classifier capable of identifying political fake news campaigns. Your task is to analyze the content of a given tweet to detect any signs that might indicate it is part of a coordinated political fake news campaign. You will be provided with the tweet text only.

    Base your decision on the text content, language use, messaging tone, and any other textual indicators of coordinated disinformation efforts. Look for signs such as sensationalism, misleading claims, highly partisan language, or attempts to provoke strong emotional responses.

    Here are some labeled examples of tweets to help guide your analysis. The labels indicate whether the tweet was part of a "Fake" coordinated political campaign or "Real" organic behavior.

    '''
    for i, example in enumerate(similar_examples, 1):
        prompt += f'''
    Example {i}:
    Tweet text: "{example['text']}"
    Label: {example['label'].capitalize()}
    '''
    
    prompt += '''
    Now, analyze the following input tweet and determine whether it suggests a "Fake" coordinated effort or "Real" organic behavior.'''
    
    return prompt

def check_astroturfing(model_name, text, similar_examples):
    prompt = generate_prompt(similar_examples)
    
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'system',
            'content': prompt
        },
        {
            'role': 'user',
            'content': f'''Tweet text: {text}

            Output:
            Fake or Real (No explanation needed).'''
        },
    ],
    options={
        "temperature": 0
    })
    return response['message']['content'].strip().lower()

def process_files(model_name, test_folder, train_folder, num_shots, result_file, vectorizer, train_vectors, train_data):
    results = load_results(result_file)
    
    # Process test files
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.json')]
    
    with tqdm(total=len(test_files), desc=f"Processing files (num_shots={num_shots})") as pbar:
        for filename in test_files:
            if filename in results:
                pbar.update(1)
                continue
            
            file_path = os.path.join(test_folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            text = extract_text(data)
            
            # Find most similar examples from train data
            similar_examples = find_most_similar(text, train_data, vectorizer, train_vectors, num_shots)
            
            prediction = check_astroturfing(model_name, text, similar_examples)
            
            save_result(results, filename, prediction, result_file)
            
            pbar.update(1)
    
    return results

def calculate_metrics(folder_path, results):
    true_labels = []
    predicted_labels = []
    
    for filename, prediction in results.items():
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        true_labels.append(1 if data['label'] == 'fake' else 0)
        predicted_labels.append(1 if prediction == 'fake' else 0)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_labels)
    
    return accuracy, precision, recall, f1, roc_auc

def main(model_size, num_shots_range):
    test_folder = './dataset1/test'  # Adjust this to your test folder path
    train_folder = './dataset1/train'  # Adjust this to your train folder path
    
    all_metrics = {}
    
    # Load and process train data once
    train_data = load_and_process_data(train_folder)
    
    # Vectorize train texts once
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform([item['text'] for item in train_data])

    if model_size=='8b':
        model_name = f'llama3.1:latest'
    else:
        model_name = f'llama3.1:{model_size}'
    
    for num_shots in num_shots_range:
        result_file = f'result_{model_size}_rag_{num_shots}_shots_no_graph.json'
        
        results = process_files(model_name, test_folder, train_folder, num_shots, result_file, vectorizer, train_vectors, train_data)
        
        # Calculate metrics
        accuracy, precision, recall, f1, roc_auc = calculate_metrics(test_folder, results)
        
        # Store metrics
        all_metrics[num_shots] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }
        
        # Print metrics
        print(f"\nResults for {num_shots}-shot RAG:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
    
    # Save all metrics to a JSON file
    with open(f'metrics_{model_size}_rag_all_shots.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

if __name__ == "__main__":
    model_size = '8b'
    num_shots_range = range(1, 9)  # This will create a range from 1 to 8
    main(model_size, num_shots_range)