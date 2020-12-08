import argparse
import numpy as np
import re
import nltk
from typing import List
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from keras import backend as K
from matplotlib import pyplot
from pathlib import Path
from tensorflow.keras.models import load_model
import sys
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

sys.path.append("E:\Slide subject\Text mining\TextSummarization\src")


from features.split_data_tokenization import tokenization
from attention import AttentionLayer

max_text_len=30
max_summary_len=8


########################################## Seq2Seq architecture ####################################################

def main():

    K.clear_session()
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--latent_dim', type=int, default=300, help="latent dimensionality of the encoding space")
    parser.add_argument("-e", '--embedding_dim', type=int, default=100, help="Embedding layer dimensionality")
    parser.add_argument("-ep", '--epochs', type=int, default=50, help="Epochs for training")
    args = parser.parse_args()



    x_tr, y_tr, x_val, y_val, x_tokenizer, y_tokenizer, x_voc, y_voc = tokenization()

    ########################################### Training phrase #####################################################
    # Encoder
    encoder_inputs = Input(shape=(max_text_len,)) # max_text_len = 30

    #embedding layer
    enc_emb =  Embedding(x_voc, args.latent_dim,trainable=True)(encoder_inputs)

    #encoder lstm 1
    encoder_lstm1 = LSTM(args.latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    #encoder lstm 2
    encoder_lstm2 = LSTM(args.latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    #encoder lstm 3
    encoder_lstm3=LSTM(args.latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))

    #embedding layer
    dec_emb_layer = Embedding(y_voc, args.embedding_dim,trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(args.latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    #dense layer
    decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile model
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    # Early stop
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
    # train model
    history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs= args.epochs,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

    path = Path(__file__).parent / "../../reports/figures/Loss_variation.png"

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig(path)
    pyplot.show()

    ########################################### Inference phrase #####################################################

    # Encode the input sequence to get the feature vector
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(args.latent_dim,))
    decoder_state_input_c = Input(shape=(args.latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_text_len,args.latent_dim))

    # Get the embeddings of the decoder sequence
    dec_emb2= dec_emb_layer(decoder_inputs)
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])

    path = Path(__file__).parent / "../../models/encoder_model_inference.h5"
    encoder_model.save(path)
    print("Encoder model for prediction has been saved ")
    path = Path(__file__).parent / "../../models/decoder_model_inference.h5"
    decoder_model.save(path)
    print("Decoder model for prediction has been saved ")
    print("traning completed")

if __name__ == "__main__":
    main()
