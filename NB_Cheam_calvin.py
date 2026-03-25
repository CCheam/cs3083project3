import numpy as np
import matplotlib.pyplot 
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


#funcs to format input files
def load_csv(filepath):
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            if len(parts) == 2:
                labels.append(parts[0].strip())
                texts.append(parts[1].strip())
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


class NaiveBayes:
    def __init__(self,training_data):
        self.training_data = training_data
        pass

    def text_tokenization(text):
        #imports stopwords, sets and tokenizes text. 
        #Filters by comparing the tokens to stopwords list
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(re.sub(r'[^a-z\s]', '', text.lower()))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        #test prints to ensure tokenization works right
        return filtered_tokens


def text_tokenization(text):
        #imports stopwords, sets and tokenizes text. 
        #Filters by comparing the tokens to stopwords list
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(re.sub(r'[^a-z\s]', '', text.lower()))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        #test prints to ensure tokenization works right
        return filtered_tokens
        

def main():

    txt_file_path = ["training_biology.txt","training_economics.txt","training_physics.txt"]
    for file in txt_file_path:
        with open(file, "r") as f:
            content = f.read()
            print(text_tokenization(content))

if __name__ == "__main__":
    main()