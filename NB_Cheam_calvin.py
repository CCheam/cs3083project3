import numpy as np
import matplotlib.pyplot 
import os 
import time 
import itertools 
import nltk
import re 
import math 

#import and setup for tokenization and stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
def text_tokenization(text):
    #imports stopwords, sets and tokenizes text. 
    #Filters by comparing the tokens to stopwords list
    stop_words = set(stopwords.words('english'))
    stop_words.add('.')
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    #test prints to ensure tokenization works right
    print("Original:", tokens)
    print("Filtered:", filtered_tokens)

    
def generate_confusion_matrix():
    mtrx = [[],[],[],[],[],[]]
    return mtrx

def generate ():
    return 1
def main():

    conf = generate_confusion_matrix()
    file_path = "training_biology.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    text_tokenization(raw_text)

if __name__ == "__main__":
    main()