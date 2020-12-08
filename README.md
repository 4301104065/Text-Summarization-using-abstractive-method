# Text Summarization
# What is Text Summarization?
Input: A corpus.
Output: a corpus has been summarized

# How to get started:
1. Clone this project: 
git clone https://github.com/4301104065/Text-Summarization-using-abtractive-method
or download this project as zip.
2. Change the working directory to TextSummarization using cd
3. Make sure python 3.7.5 is installed in your computer
4. Install all the requirements library, package by using pip: pip install -r requirements.txt
5. Run setup.py to install necessary nltk packages: python setup.py

# Folder Structure:
1. data:
data/raw: Store many samples containing original texts and summaried texts
data/processed: Store file data_processed has been processed manually through many steps like lowercasing, removing 's and other preprocessings.

2. models:
Containing model which has been trained through seq2seq architecture using encoder and decoder model.
Folder including 2 model : decoder_model_inference.h5 and encoder_model_inference.h5
3. reports:
Including file word,ppt,pdf illustrating this project. Besides, this folder also contains images demonstrating data distribution and loss_variation in traning process.
4. src: 
containing scripts.
src/data: containing file Processing_data.py for preprocessing data
src/features: containing file split_data_tokenization.py  for splitting processed data and tokenization process 
src/model: Containing three files: attention.py serving for attention mechanism used for training process, train_model.py used for training model seq2seq and save informative encoder_model_inference and decoder_model_inference in file .h5. File predict_model.py used for predicting unknow sample in validation set.
5. venv: 
containing virtual environment parts.
# How to preprocess a raw corpus :
0. Make sure the working directory has been set to main directory (cd TextSummarization)
1. Place a raw corpus in directory data/raw.
2. In CLI: python src/data/processing_data.py
# How to visualize the overview of dataset
0. Make sure the working directory has been set to main directory
1. In CLI: python src/visualization/visualize.py
# How to train a seq2seq model with specific parameters:
0. Make sure the working directory has been set to main directory (cd TextSummarization)
1. In CLI: python src/modesl/train_model.py -l 300 -e 100 -ep 50
- l : latent_dim ( latent dimensionality of encoding space), i set default args which is 300
- e: embedding dimensionality, i set default args which is 100
- ep: epochs, i set default args which is 50
2. After the model has been trained. It will be save to directory TextSummarization/models.
# How to predict a topic of raw corpus text files:
0. Make sure the working directory has been set to main directory (cd TextSummarization)
1. In CLI: python src/model/predict_model.py ( this file predicted 10 first samples in the validation set through encoder_model_inference.h5 and decoder_model_inference.h5 having saved previously. 
