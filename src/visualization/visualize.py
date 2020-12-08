import argparse
import numpy as np
import re
import nltk
from typing import List
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd





def main():

    text_word_count = []
    summary_word_count = []

    path = Path(__file__).parent / "../../data/processed/data_processed.csv"
    data = pd.read_csv(path)

    # populate the lists with sentence lengths
    for i in data['Text']:
        text_word_count.append(len(i.split()))

    for i in data['Summary']:
        summary_word_count.append(len(i.split()))

    length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})

    length_df.hist(bins = 30)
    path = Path(__file__).parent / "../../reports/figures/distribtion.png"
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()
