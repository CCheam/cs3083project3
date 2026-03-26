import numpy as np
import matplotlib.pyplot as plt
import os 
import time 
import itertools 
import nltk
import re 
import math 
"""
leaning on https://www.geeksforgeeks.org/machine-learning/naive-bayes-scratch-implementation-using-python/ for help in this
"""
#import and setup for tokenization and stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
"""nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')"""
outside_data = 'https://github.com/Gituhin/Sentence-Classification-naive-bayes-/blob/main/traindata.csv'

#file assist funcs
def normalize_label(label):
    """
    Standardize label strings so training and test sets use the same names.
    e.g. 'world history' -> 'history', 'World History' -> 'history'
    """
    label = label.lower().strip()
    mapping = {
        'world history': 'history',
        'hist':          'history',
        'bio':           'biology',
        'econ':          'economics',
        'phys':          'physics',
    }
    return mapping.get(label, label)

#funcs to format input files
def load_csv(filepath):
    # 0 label, 1 us text
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            if len(parts) == 2:
                labels.append(normalize_label(parts[0].strip()))
                texts.append(parts[1].strip())
    return texts, labels

def load_colon_separated(filepath):
   # file func for training data
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            idx = line.index(':')
            label = normalize_label(line[:idx].strip())
            text  = line[idx+1:].strip()
            labels.append(label)
            texts.append(text)
    return texts, labels

def load_txt_files(folder_path):
    texts, labels = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            category = filename.replace('.txt', '')
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
                        labels.append(category)
    return texts, labels

def load_named_txt_files(file_list):
    """
    Load specific named .txt files from the current directory.
    Derives the label from the filename (e.g. 'training_biology.txt' → 'biology').
    """
    texts, labels = [], []
    for filepath in file_list:
        # Extract a clean category name from the filename
        basename = os.path.basename(filepath)
        category = basename.replace('.txt', '')
        # Strip common prefixes like 'training_' or 'testing_'
        for prefix in ('training_', 'testing_'):
            if category.startswith(prefix):
                category = category[len(prefix):]
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(category)
    return texts, labels

class NaiveBayes:
    def __init__(self,alpha=2.0,remove_stopwords=True):
        self.alpha=alpha
        self.remove_stopwords=remove_stopwords
        self.word_counts = {}         # count of each word per category
        self.total_words = {}         # total words per category
        self.vocab = set()            # absolute unique words across all text
        self.classes = []             # list of unique categories
        self.log_priors = {}
        pass

    def train(self, texts, labels):
        # Setup category dictionaries
        self.classes = list(set(labels))
        total_docs = len(labels)
        
        for c in self.classes:
            self.word_counts[c] = {}
            self.total_words[c] = 0
            # Calculate Priors 
            doc_count = labels.count(c)
            self.log_priors[c] = math.log(doc_count / total_docs)
            
        # Count frequencies per class
        for text, label in zip(texts, labels):
            # Pass the setting to the tokenization function
            tokens = text_tokenization(text, remove_stopwords=self.remove_stopwords)
            for word in tokens:
                self.vocab.add(word)
                self.total_words[label] += 1
                self.word_counts[label][word] = self.word_counts[label].get(word, 0) + 1

    def predict(self, text):
        # Use the class setting for stopwords
        tokens = text_tokenization(text, remove_stopwords=self.remove_stopwords)
        class_scores = {}
        vocab_size = len(self.vocab)
        
        for c in self.classes:
            score = self.log_priors[c]
            for word in tokens:
                # Laplace Smoothing using self.alpha
                count_w_c = self.word_counts[c].get(word, 0)
                word_prob = (count_w_c + self.alpha) / (self.total_words[c] + (self.alpha * vocab_size))
                score += math.log(word_prob)
            class_scores[c] = score
            
        return max(class_scores, key=class_scores.get)

    def predict_all(self, texts):
        #Predict labels for a list of text
        return [self.predict(t) for t in texts]

"""Evaluation funcs"""
def compute_accuracy(true_labels, predicted_labels):
    """Return the fraction of predictions that match the true labels."""
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    return correct / len(true_labels)
 
 
def compute_confusion_matrix(true_labels, predicted_labels, classes):
    """
    Build a confusion matrix as a 2-D numpy array.
    Row = true class, Column = predicted class (both ordered by `classes`).
    """
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        if true in idx and pred in idx:
            matrix[idx[true]][idx[pred]] += 1
    return matrix

def plot_confusion_matrix(matrix, classes, title='Confusion Matrix', save_path=None):
    """
    Plot the confusion matrix as a colour-coded heat-map using matplotlib.
    Saves to `save_path` if provided; otherwise calls plt.show().
    """
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.5), max(5, len(classes) * 1.3)))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
 
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
 
    thresh = matrix.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]),
                    ha='center', va='center',
                    color='white' if matrix[i, j] > thresh else 'black')
 
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  → Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    plt.close()

 
def find_misclassified(texts, true_labels, pred_labels, n=3):
    """
    Return up to `n` misclassified examples as a list of
    (text, true_label, predicted_label) tuples.
    """
    examples = []
    for text, true, pred in zip(texts, true_labels, pred_labels):
        if true != pred:
            examples.append((text, true, pred))
        if len(examples) >= n:
            break
    return examples
 
def print_confusion_matrix(matrix, classes):
    """Print the confusion matrix to the console."""
    col_width = max(len(c) for c in classes) + 2
    header = ' ' * col_width + ''.join(c.ljust(col_width) for c in classes)
    print("\nConfusion Matrix (rows=True, cols=Predicted):")
    print(header)
    for i, c in enumerate(classes):
        row_str = c.ljust(col_width) + ''.join(str(matrix[i][j]).ljust(col_width) for j in range(len(classes)))
        print(row_str)

def text_tokenization(text, remove_stopwords):
    stop_words = set(stopwords.words('english'))
    # Clean and tokenize
    tokens = word_tokenize(re.sub(r'[^a-z\s]', '', text.lower()))
    
    if remove_stopwords:
        return [word for word in tokens if word not in stop_words]
    return tokens
        
def run_experiment(train_texts, train_labels, test_texts, test_labels,
                   alpha=1.0, remove_stopwords=True, label=""):
    """
    Train a NaiveBayes model, predict on the test set, and return
    (model, predictions, accuracy, elapsed_seconds).
    """
    model = NaiveBayes(alpha=alpha, remove_stopwords=remove_stopwords)
    start = time.time()
    model.train(train_texts, train_labels)
    preds = model.predict_all(test_texts)
    elapsed = time.time() - start
    acc = compute_accuracy(test_labels, preds)
    tag = f"[{label}] " if label else ""
    print(f"{tag}alpha={alpha}, stopwords_removed={remove_stopwords}  "
          f"→  accuracy={acc:.4f}  ({elapsed:.2f}s)")
    return model, preds, acc, elapsed
 
 
def plot_alpha_sweep(train_texts, train_labels, test_texts, test_labels,
                     alphas=None, title="Accuracy vs Alpha", save_path=None):
    """
    Train models with a range of alpha (smoothing) values and plot accuracy.
    """
    if alphas is None:
        alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    accuracies = []
    for a in alphas:
        model = NaiveBayes(alpha=a)
        model.train(train_texts, train_labels)
        preds = model.predict_all(test_texts)
        accuracies.append(compute_accuracy(test_labels, preds))
 
    plt.figure(figsize=(8, 4))
    plt.plot(alphas, accuracies, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha (Laplace smoothing)')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  → Alpha sweep plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()
 
 
# ─────────────────────────────────────────────
# COMBINED DATASET: TXT + CSV TRAINING
# ─────────────────────────────────────────────
 
def run_combined_dataset():
    """
    Train on the provided .txt training files AND the extra CSV dataset combined,
    then evaluate on the provided test file.
 
    Required files in the working directory:
        training_biology.txt, training_economics.txt,
        training_physics.txt, training_history.txt   ← training data (txt)
        extra_dataset.csv                             ← extra training data (CSV)
        testing_data.txt / testing_data.csv           ← test set (labeled)
 
    The model sees all txt + all csv samples during training.
    Evaluation is performed on the separate test file only.
    """
    print("\n" + "=" * 60)
    print("COMBINED TRAINING – TXT files + Extra CSV dataset")
    print("=" * 60)
 
    # ── Load TXT training data ──
    train_files = [
        "training_biology.txt",
        "training_economics.txt",
        "training_physics.txt",
        "training_history.txt",
    ]
    missing_txt = [f for f in train_files if not os.path.exists(f)]
    if missing_txt:
        print(f"  ⚠  Missing txt training files: {missing_txt}")
        txt_texts, txt_labels = [], []
    else:
        txt_texts, txt_labels = load_named_txt_files(train_files)
        print(f"  TXT training samples: {len(txt_texts)}")
        for c in sorted(set(txt_labels)):
            print(f"    {c}: {txt_labels.count(c)}")
 
    # ── Load CSV training data ──
    csv_file = "extra_dataset.csv"
    #data is extracted raw: https://github.com/Gituhin/Sentence-Classification-naive-bayes-/blob/main/traindata.csv
    csv_texts, csv_labels = load_csv(csv_file)
    print(f"  CSV training samples: {len(csv_texts)}")
    for c in sorted(set(csv_labels)):
        print(f"    {c}: {csv_labels.count(c)}")
 
    # Merge sources
    train_texts = txt_texts + csv_texts
    train_labels = txt_labels + csv_labels
 
    if not train_texts:
        print("  ✗  No training data found. Aborting.")
        return
 
    print(f"\n  Combined training set: {len(train_texts)} samples across "
          f"{len(set(train_labels))} classes: {sorted(set(train_labels))}")
 
    # ── Load test data ──
    test_file = "testing_data.txt"
    if not os.path.exists(test_file):
        test_file = "testing_data.csv"
    if not os.path.exists(test_file):
        print("  ⚠  No test file found (testing_data.txt / testing_data.csv). Aborting.")
        return
 
    if test_file.endswith('.csv'):
        test_texts, test_labels = load_csv(test_file)
    else:
        # testing_data.txt uses "label: text" format on each line
        test_texts, test_labels = load_colon_separated(test_file)
 
    print(f"  Test samples: {len(test_texts)}")
 
    # ── Baseline experiment ──
    print("\n--- Baseline (alpha=1, stopwords removed) ---")
    model, preds, acc, _ = run_experiment(
        train_texts, train_labels, test_texts, test_labels,
        alpha=1.0, remove_stopwords=True, label="combined-baseline"
    )
 
    # ── Confusion matrix ──
    # Use only classes that appear in the test set so the matrix stays readable
    classes = sorted(set(test_labels))
    cm = compute_confusion_matrix(test_labels, preds, classes)
    print_confusion_matrix(cm, classes)
    plot_confusion_matrix(cm, classes,
                          title="Combined Training – Confusion Matrix",
                          save_path="confusion_matrix_combined.png")
 
    # ── Misclassified examples ──
    print("\n--- Misclassified Examples ---")
    examples = find_misclassified(test_texts, test_labels, preds, n=5)
    if examples:
        for text, true, pred in examples:
            print(f"  True={true:15s}  Pred={pred:15s}  → \"{text[:80]}\"")
    else:
        print("  No misclassifications found!")
 
    # ── Preprocessing ablation ──
    print("\n--- Preprocessing Ablation (no stopword removal) ---")
    run_experiment(train_texts, train_labels, test_texts, test_labels,
                   alpha=1.0, remove_stopwords=False, label="combined-no-stopwords")
 
    # ── Alpha sweep ──
    print("\n--- Alpha Sweep ---")
    plot_alpha_sweep(train_texts, train_labels, test_texts, test_labels,
                     title="Combined Training – Accuracy vs Alpha",
                     save_path="alpha_sweep_combined.png")
    

def main():
    #tokenization test
    sample = "The mitochondria is the powerhouse of the cell!"
    print(f"  Input : {sample}")
    print(f"  Tokens: {text_tokenization(sample,True)}")
    run_combined_dataset()
    
if __name__ == "__main__":
    main()