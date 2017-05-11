import numpy as np 
import pandas as pd
import os, copy

from keras.preprocessing import text
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.models import Sequential

from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.optimizers import RMSprop
from keras.engine.topology import Merge

train_df = pd.read_csv('train.csv',encoding='utf-8').fillna("")
train_df.head()

X = train_df[['question1','question2']]
y = train_df['is_duplicate']

N = len(X)

X_train = X[:int(N/2)]
X_test = X[int(N/2):]
y_train = y[:int(N/2)]
t_test = y[int(N/2):]
X_train.head()

q1s = list(X_train['question1'].apply(lambda x: x.encode('utf-8'))) 
q2s = list(X_train['question2'].apply(lambda x: x.encode('utf-8')))
all_questions = q1s + q2s

tok = text.Tokenizer()
tok.fit_on_texts(all_questions)

q1s_tok = tok.texts_to_sequences(q1s)
q2s_tok = tok.texts_to_sequences(q2s)


maxlen = max([len(i) for i in q1s_tok+q2s_tok])
maxlen

#load glove embedding
embeddings_index = {}
f = open(os.path.join('../word2vec/', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

EMBEDDING_DIM = 100

print('Found %s word vectors.' % len(embeddings_index))

word_index = tok.word_index

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#now the NN stuff

from keras.preprocessing import sequence
from keras.models import Sequential

from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.optimizers import RMSprop
from keras.engine.topology import Merge
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization


q1s_tok = sequence.pad_sequences(q1s_tok,maxlen=maxlen)
q2s_tok = sequence.pad_sequences(q2s_tok,maxlen=maxlen)

filter_length = 5
nb_filter = 64
pool_length = 4

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False))

#model1.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
#model1.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model1.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model1.add(GlobalMaxPooling1D())
#model1.add(Dropout(0.1))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False))
model2.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model2.add(GlobalMaxPooling1D())
#model2.add(Dropout(0.1))
#model2.add(LSTM(128, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
#model2.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))


merged_model = Sequential()
merged_model.add(Merge([model1,model2],mode='concat'))
#merged_model.add(Dense(128))
merged_model.add(Dense(1))

merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

merged_model.fit([q1s_tok,q2s_tok], y=y_train, batch_size=64, nb_epoch=10,
                 verbose=1, shuffle=True, validation_split=0.1)

#print training probs
probs=merged_model.predict_proba([q1s_tok,q2s_tok])

pd.DataFrame(probs).to_csv('probs_train.csv',header=False)

#print test probs
q1s = list(X_test['question1'].apply(lambda x: x.encode('utf-8'))) 
q2s = list(X_test['question2'].apply(lambda x: x.encode('utf-8')))
q1s_tok = tok.texts_to_sequences(q1s)
q2s_tok = tok.texts_to_sequences(q2s)
q1s_tok = sequence.pad_sequences(q1s_tok,maxlen=maxlen)
q2s_tok = sequence.pad_sequences(q2s_tok,maxlen=maxlen)

probs=merged_model.predict_proba([q1s_tok,q2s_tok])
pd.DataFrame(probs).to_csv('probs_test.csv',header=False)

merged_model.save('quora_glove.h5')