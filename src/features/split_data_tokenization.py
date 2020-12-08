import numpy as np
import re
import nltk
from typing import List
from sklearn.datasets import load_files
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd



def tokenization():

        max_text_len=30
        max_summary_len=8

        # Read the processed dataset
        path = Path(__file__).parent / "../../data/processed/data_processed.csv"
        data = pd.read_csv(path)

        x_tr,x_val,y_tr,y_val=train_test_split(np.array(data['Text']), np.array(data['Summary']),test_size=0.1,random_state=0,shuffle=True)


        # a) text tokenizer

        x_tokenizer = Tokenizer()
        x_tokenizer.fit_on_texts(list(x_tr))



        # Here, I am defining the threshold to be 6 which means word whose count is below 6 is considered as a rare word
        thresh=6

        cnt=0
        tot_cnt=0
        freq=0
        tot_freq=0

        for key,value in x_tokenizer.word_counts.items():
            tot_cnt=tot_cnt+1
            tot_freq=tot_freq+value
            if(value<thresh):
                cnt=cnt+1
                freq=freq+value


        # tot_cnt gives the size of vocabulary (which means every unique words in the text)

        # cnt gives me the no. of rare words whose count falls below threshold

        # tot_cnt - cnt gives me the top most common words

        # prepare a tokenizer for reviews on training data
        x_tokenizer = Tokenizer(num_words=tot_cnt-cnt) # retain the most used words in data
        x_tokenizer.fit_on_texts(list(x_tr))

        # convert text sequences into integer sequences
        x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
        x_val_seq = x_tokenizer.texts_to_sequences(x_val)

        # padding zero upto maximum length
        x_tr = pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
        x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

        # size of vocabulary ( +1 for padding token)
        x_voc = x_tokenizer.num_words + 1

        # b) Summary tokenizer

        y_tokenizer = Tokenizer()
        y_tokenizer.fit_on_texts(list(y_tr))

        # tot_cnt gives the size of vocabulary (which means every unique words in the text)

        # cnt gives me the no. of rare words whose count falls below threshold

        # tot_cnt - cnt gives me the top most common words

        # Here, I am defining the threshold to be 6 which means word whose count is below 6 is considered as a rare word
        for key,value in x_tokenizer.word_counts.items():
            tot_cnt=tot_cnt+1
            tot_freq=tot_freq+value
            if(value<thresh):
                cnt=cnt+1
                freq=freq+value


        # Let us define the tokenizer with top most common words for reviews.
        # prepare a tokenizer for reviews on training data
        y_tokenizer = Tokenizer(num_words=tot_cnt-cnt)  # retain the most used words in data
        y_tokenizer.fit_on_texts(list(y_tr))

        # convert text sequences into integer sequences
        y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr)
        y_val_seq   =   y_tokenizer.texts_to_sequences(y_val)

        # padding zero upto maximum length
        y_tr    =   pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
        y_val   =   pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

        # size of vocabulary
        y_voc = y_tokenizer.num_words +1

        # I am deleting the rows that contain only START and END tokens and not include other words
        ind=[]
        for i in range(len(y_tr)):
            cnt=0
            for j in y_tr[i]:
                if j!=0:
                    cnt=cnt+1
            if(cnt==2):
                ind.append(i)

        y_tr = np.delete(y_tr,ind, axis=0)
        x_tr = np.delete(x_tr,ind, axis=0)

        # I am deleting the rows that contain only START and END tokens and not include other words
        ind=[]
        for i in range(len(y_val)):
            cnt=0
            for j in y_val[i]:
                if j!=0:
                    cnt=cnt+1
            if(cnt==2):
                ind.append(i)

        y_val =np.delete(y_val,ind, axis=0)
        x_val =np.delete(x_val,ind, axis=0)

        return x_tr, y_tr, x_val, y_val, x_tokenizer, y_tokenizer, x_voc, y_voc

