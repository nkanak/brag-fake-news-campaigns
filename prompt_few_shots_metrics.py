import json
import os
from collections import defaultdict
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import networkx as nx
import statistics

RESULT_FILE = 'result_six_shots_graph_metrics.json'

def extract_text_and_metrics(data):
    text = data['nodes'][0]['tweet_text']
    G = nx.Graph()
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'])
    
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Calculate median retweet time using 'delay'
    retweet_times = [node['delay'] for node in data['nodes'][1:]]  # Exclude original tweet
    median_retweet_time = statistics.median(retweet_times) if retweet_times else 0
    
    metrics = {
        'avg_degree_centrality': sum(degree_centrality.values()) / len(degree_centrality),
        'eigenvector_centrality': sum(eigenvector_centrality.values()) / len(eigenvector_centrality),
        'median_retweet_time': median_retweet_time
    }
    
    return text, metrics

def check_astroturfing(text, metrics):
    response = ollama.chat(model='llama3.1:70b', messages=[
        {
            'role': 'system',
            'content': '''You are an intelligent classifier capable of identifying political fake news campaigns. Your task is to analyze the propagation of a given tweet, focusing on aggregated network metrics to detect any coordinated behavior that might indicate a political fake news campaign. You will be provided with the tweet description and a set of statistics derived from its retweet propagation graph. These statistics summarize key structural and behavioral aspects of the network, such as the average degree centrality, eigenvector centrality, clustering coefficient, and retweet timing patterns.

            The following centrality metrics are important for your analysis:

            Degree centrality measures how many direct connections a user (node) has. A high degree centrality indicates that the user is directly connected to many others, suggesting they play a central role in propagating the tweet. In a coordinated campaign, several users with unusually high degree centrality may indicate organized amplification.

            Eigenvector centrality measures the influence of a user within the network. It considers not only the number of connections a user has, but also the importance of those connections. A user with high eigenvector centrality is connected to other highly influential users. In coordinated behavior, high eigenvector centrality across specific users may indicate a core group driving the retweet campaign.

            Retweet timing patterns refer to how the retweets are distributed over time. In coordinated campaigns, retweets often occur in rapid bursts, indicating synchronized activity. In contrast, organic propagation typically shows more dispersed retweet times.

            Base your decision also on the text content, language use, and messaging tone.

            Additionally, here are some labeled examples of tweet propagations to help guide your analysis. The labels indicate whether the tweet was part of a "Fake" coordinated political campaign or "Real" organic behavior.

            Example 1:
            Tweet text: "Lady Gaga's 'The Cure' is Platinum eligible in the US. @Interscope should get it certified by @RIAA in celebration… https://t.co/r2J9Tnm0ZF"
            Propagation statistics:
            Average degree centrality: 0.05
            Eigenvector centrality: 0.70
            Median retweet time: 12627.00 seconds
            Label: Fake
                
            Example 2:
            Tweet text: "CNN host Fareed Zakaria calls for jihad rape of white women https://t.co/2jELVuFj10"
            Propagation statistics:
            Average degree centrality: 0.18
            Eigenvector centrality: 0.58
            Median retweet time: 422.00 seconds
            Label: Fake
                
            Example 3:
            Tweet text: "US House endorsements from Revolution LA! We are proud to endorse these progressive warriors to represent the peo… https://t.co/WzwCNlVW21"
            Propagation statistics:
            Average degree centrality: 0.05
            Eigenvector centrality: 0.55
            Median retweet time: 13740.00 seconds
            Label: Fake
                
            Example 4:
            Tweet text: "Axios has a first look at a new 10-page debate memo drafted by Ron Klain, who has been a debate guru for every Demo… https://t.co/rzSVNkEm46"
            Propagation statistics:
            Average degree centrality: 0.10
            Eigenvector centrality: 0.70
            Median retweet time: 7053.00 seconds
            Label: Real
                
            Example 5:
            Tweet text: "@realDonaldTrump White House deletes Putin's acknowledgement that he ordered his people to assist Trump during elec… https://t.co/WcHxOhGcW6"
            Propagation statistics:
            Average degree centrality: 0.22
            Eigenvector centrality: 0.57
            Median retweet time: 4344.00 seconds
            Label: Real
                
            Example 6:
            Tweet text: "There are moments in a rock star's life that define who he is.  🚀@TaronEgerton stars as Sir Elton John (… https://t.co/inqsGo6EZC"
            Propagation statistics:
            Average degree centrality: 0.10
            Eigenvector centrality: 0.69
            Median retweet time: 705311.00 seconds
            Label: Real

            Now, analyze the following input and determine whether the tweet's propagation suggests a "Fake" coordinated effort or "Real" organic behavior.'''
        },
        {
            'role': 'user',
            'content': f'''Tweet text: {text}
            Propagation statistics:
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

def load_results():
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_result(results, filename, prediction):
    results[filename] = prediction
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def process_files(folder_path):
    results = load_results()
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    with tqdm(total=len(json_files), desc="Processing files") as pbar:
        for filename in json_files:
            if filename in results:
                pbar.update(1)
                continue
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            text, metrics = extract_text_and_metrics(data)
            prediction = check_astroturfing(text, metrics)
            
            save_result(results, filename, prediction)
            
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
    
    return accuracy, precision, recall, f1

def main():
    folder_path = './dataset1/test'  # Adjust this to your test folder path
    results = process_files(folder_path)
    
    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(folder_path, results)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()