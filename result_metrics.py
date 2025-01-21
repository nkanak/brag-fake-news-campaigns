import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def extract_label(text):
    """
    Extract label from text, converting 'fake', 'real', 'Yes', and 'No' to appropriate labels.
    """
    if isinstance(text, str):
        text = text.lower()
        if 'fake' in text or text == 'yes':
            return 'fake'
        elif 'real' in text or text == 'no':
            return 'real'
    return text

def load_true_labels(test_folder):
    """
    Load true labels from JSON files in the test folder.
    """
    true_labels = {}
    for filename in os.listdir(test_folder):
        if filename.endswith('.json'):
            with open(os.path.join(test_folder, filename), 'r') as f:
                data = json.load(f)
                true_labels[filename] = data['label']
    return true_labels

def load_predicted_labels(prediction_file):
    """
    Load predicted labels from a JSON file and extract labels.
    """
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    return {k: extract_label(v) for k, v in predictions.items()}

def convert_labels_to_binary(labels):
    """
    Convert labels to binary format (1 for 'fake' or 'Yes', 0 for 'real' or 'No').
    """
    return [1 if label in ['fake', 'yes'] else 0 for label in labels]

def calculate_metrics(y_true, y_pred):
    """
    Calculate various evaluation metrics including FPR and TPR.
    """
    y_true_binary = convert_labels_to_binary(y_true)
    y_pred_binary = convert_labels_to_binary(y_pred)
    
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    roc_auc = roc_auc_score(y_true_binary, y_pred_binary)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    # Calculate FPR and TPR
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)  # This is the same as recall
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'tpr': tpr,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'fpr': fpr
    }

def main():
    """
    Main function to load data, calculate metrics, and print results.
    """
    test_folder = './dataset1/test'
    prediction_file = './result_rag_k-means_5_shots_no_graph.json'
    
    true_labels = load_true_labels(test_folder)
    predicted_labels = load_predicted_labels(prediction_file)
    
    # Ensure we're comparing the same files
    common_files = set(true_labels.keys()) & set(predicted_labels.keys())
    
    y_true = [true_labels[file] for file in common_files]
    y_pred = [predicted_labels[file] for file in common_files]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()