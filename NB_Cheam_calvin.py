import numpy as np
import matplotlib.pyplot 
import os 
import time 
import itertools 
import nltk
import re 
import math 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def normalize_text(text):
    ft = text.lower()
    for char in ft:
        if not char.isalnum():
            #remove
            print()
    return ft
    
def generate_confusion_matrix():
    mtrx = [[],[],[],[],[],[]]
    return mtrx

def generate ():
    return 1
def main():

    conf = generate_confusion_matrix()
    for row in conf:
        print(row)
    return -1
if __name__ == "__main__":
    main()