import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams

eng_stopwords = set(stopwords.words('english'))

from keras.preprocessing import text
from sklearn.model_selection import train_test_split

test_df = pd.read_csv('test_unknowns.csv',encoding='utf-8').fillna("")

X = test_df[['q1','q2']]

q1s = list(X['q1'].apply(lambda x: x.encode('utf-8'))) 
q2s = list(X['q2'].apply(lambda x: x.encode('utf-8')))
all_questions = q1s + q2s

import pickle

with open('tokenizer.pickle') as f:
	tok = pickle.load(f)

q1s_tok = tok.texts_to_sequences(q1s)
q2s_tok = tok.texts_to_sequences(q2s)

#now the NN stuff

from keras.preprocessing import sequence
from keras.models import Sequential, load_model

from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.optimizers import RMSprop
from keras.engine.topology import Merge


q1s_tok = sequence.pad_sequences(q1s_tok,maxlen=205)
q2s_tok = sequence.pad_sequences(q2s_tok,maxlen=205)

model = load_model('quora.h5')
probs = model.predict_proba([q1s_tok,q2s_tok])

df = pd.DataFrame(probs)

df.to_csv('predictions.csv')