import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams

from keras.preprocessing import text
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('training_unknowns.csv',encoding='utf-8').fillna("")
train_df.head()

X = train_df[['question1_uk','question2_uk']]
y = train_df['is_duplicate']

#X_train, X_test, y_train, y_test = train_test_split(X,y)
#X_train.head()

X_train = X.copy()
y_train = y.copy() #use the whole set

q1s = list(X_train['question1_uk'].apply(lambda x: x.encode('utf-8'))) 
q2s = list(X_train['question2_uk'].apply(lambda x: x.encode('utf-8')))
all_questions = q1s + q2s

tok = text.Tokenizer(nb_words=6000)
tok.fit_on_texts(all_questions)

#import pickle
#pickle.dump(tok,open('tokenizer.pickle','w'))

q1s_tok = tok.texts_to_sequences(q1s)
q2s_tok = tok.texts_to_sequences(q2s)

q1s_tok[0]

maxlen = max([len(i) for i in q1s_tok+q2s_tok])
maxlen

#now the NN stuff

from keras.preprocessing import sequence
from keras.models import Sequential

from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.optimizers import RMSprop
from keras.engine.topology import Merge

#define log-loss function
from keras import backend as K

def log_loss(y_true,y_pred):
	first_log = K.log(K.clip(y_pred, K.epsilon(), None))
	second_log = K.log(K.clip(1.-y_pred, K.epsilon(), None))
	return K.mean(-y_true * first_log - (1. - y_true) * second_log)

yt = np.array([1,0,1])
yp = np.array([0.99,0.01,0.01])




q1s_tok = sequence.pad_sequences(q1s_tok,maxlen=205)
q2s_tok = sequence.pad_sequences(q2s_tok,maxlen=205)

model1 = Sequential()
model1.add(Embedding(6000, 128, dropout=0.1))
model1.add(LSTM(128, dropout_W=0.3, dropout_U=0.3, return_sequences=True))
model1.add(LSTM(128, dropout_W=0.3, dropout_U=0.3, return_sequences=True))
model1.add(LSTM(128, dropout_W=0.3, dropout_U=0.3))

model2 = Sequential()
model2.add(Embedding(6000, 128, dropout=0.1))
model2.add(LSTM(128, dropout_W=0.3, dropout_U=0.3, return_sequences=True))
model2.add(LSTM(128, dropout_W=0.3, dropout_U=0.3, return_sequences=True))
model2.add(LSTM(128, dropout_W=0.3, dropout_U=0.3))

merged_model = Sequential()
merged_model.add(Merge([model1,model2],mode='concat'))
merged_model.add(Dense(1))

merged_model.add(Activation('sigmoid'))

merged_model.compile(loss=log_loss, optimizer='adam', metrics=['accuracy'])

merged_model.fit([q1s_tok,q2s_tok], y=y_train, batch_size=64, nb_epoch=20,
                 verbose=1, shuffle=True, validation_split=0.1)

merged_model.save('quora.h5')