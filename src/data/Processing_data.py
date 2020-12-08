
import numpy as np
import re
import nltk
from typing import List
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pathlib import Path
from bs4 import BeautifulSoup

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                      "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                      "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                      "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                      "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                      "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                      "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                      "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                      "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                      "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                      "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                      "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                      "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                      "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                      "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                      "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                      "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                      "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                      "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                      "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                      "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                      "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                      "you're": "you are", "you've": "you have"}

# We will perform the below preprocessing tasks for our data:

# 1.Convert everything to lowercase

# 2.Remove HTML tags

# 3.Contraction mapping

# 4.Remove (‘s)

# 5.Remove any text inside the parenthesis ( )

# 6.Eliminate punctuations and special characters

# 7.Remove stopwords

# 8.Remove short words

stop_words = set(stopwords.words('english'))

def text_cleaner(text,num):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:                                                 #removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip()


def main():
        # Read the dataset
        path = Path(__file__).parent / "../../data/raw/Reviews.csv"
        data = pd.read_csv(path, nrows=100000)

        # Drop duplicates and Na values
        data.drop_duplicates(subset=['Text'],inplace=True) #dropping duplicates
        data.dropna(axis=0,inplace=True) #dropping na

        # Filter columns
        data = data[['Text','Summary']]
        # Preprocessing

        # a) Text cleaning
        cleaned_text = []
        for t in data['Text']:
            cleaned_text.append(text_cleaner(t,0))

        # b) Summary cleaning
        cleaned_summary = []
        for t in data['Summary']:
            cleaned_summary.append(text_cleaner(t,1))

        df = pd.DataFrame()
        df['Text'] = cleaned_text
        df['Summary'] = cleaned_summary

        # Drop empty rows
        df.replace('', np.nan, inplace=True)
        df.dropna(axis=0,inplace=True)

        # Setting max length of text and summary depended on data analysis in file jupyter notebook
        max_text_len = 30
        max_summary_len = 8

        cleaned_text =np.array(df['Text'])
        cleaned_summary=np.array(data['Summary'])

        short_text=[]
        short_summary=[]

        # Extract data which its length < max len
        for i in range(len(cleaned_text)):
            if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
                short_text.append(cleaned_text[i])
                short_summary.append(cleaned_summary[i])

        df=pd.DataFrame({'Text':short_text,'Summary':short_summary})

        # Add token sostok (_star_) and eostok(_end_) in the summary
        df['Summary'] = df['Summary'].apply(lambda x : 'sostok '+ x + ' eostok')

        output = Path(__file__).parent / "../../data/processed/data_processed.csv"
        df.to_csv(path_or_buf = output)
        print("\n")
        print('Processed Data Saved')
        print("Data after processing: ")
        print("\n")
        for i in range (5):
            print("Review: ",df['Text'][i])
            print("Sumarry: ",df['Summary'][i])
            print("\n")

        print("Remember sostok = _start_ and eostok = _end_")

if __name__ == "__main__":
    main()
