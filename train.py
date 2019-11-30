import numpy as np
import pandas as pd
import re
import string

# Function to to do some basic cleanup
def clean_all(text):
    text = text.lower()  # make lowercase
    text = text.encode('ascii', 'ignore').decode('ascii')  # remove emoji characters
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuation
    return text


filename = input('Enter csv filename for training: ')
df = pd.read_csv(filename)


# Load Glove model, publicly available
file = open("glove.6B.300d.txt", encoding="utf8")

# Save in dictionary
word_vecs = {}
for line in file:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], "float32")
    word_vecs[word] = vector
    
# Assign every feature in the dataeet to set of vectors
# Might be a bit complicated here, essentially just incorporates words not exisiting in the pre-trained model as well
feature_vecs = []
for feature in df.cleaned_feat:
    lst = feature.split()
    num_words = len(lst)
    feature_vec = np.zeros((1, 300))
    for word_n in range(num_words):
        word = lst[word_n]
        if word in word_vecs:
            x = word_vecs[word]
            x = np.expand_dims(np.array(x), axis=-1)
            feature_vec = np.concatenate((feature_vec, x.T))
        else:
            word_splitted = [char for char in word]
            for word2 in word_splitted:
                x = word_vecs[word2]
                x = np.expand_dims(np.array(x), axis=-1)
                feature_vec = np.concatenate((feature_vec, x.T))
    feature_vec = feature_vec[1:, :]
    feature_vecs.append(feature_vec)
feature_vecs = np.array(feature_vecs)

# Compress all the features to consistent vectors for training
num_features = df.shape[0]
num_categories = 3
words_dim = 300
feature_vecs_train = np.zeros((num_features, num_categories, words_dim))

# Use singular value decomposition to get the three strongest features from all feature vectors
for vec_n in range(len(feature_vecs)):
    vec = feature_vecs[vec_n]
    u, s, vh = np.linalg.svd(vec, full_matrices=True)
    vec_reduced = vh[0:3, :]
    feature_vecs_train[vec_n, :, :] = vec_reduced
feature_vecs_train = np.reshape(feature_vecs_train, newshape=(df.shape[0], -1))


# Function to assign all values to in training vectors between 0 and 1
def scale_array(dat, out_range=(0, 1)):
    domain = [np.min(dat, axis=0), np.max(dat, axis=0)]

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b =  1.0 / domain[1]
        return (x - domain[0]) / b
        
    return interp(uninterp(dat))

for i in range(feature_vecs_train.shape[0]):
    feature_vecs_train[i] = scale_array(feature_vecs_train[i])
    
    
for i in range(feature_vecs_train.shape[0]):
    feature_vecs_train[i] = scale_array(feature_vecs_train[i])
    
    
    
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(200, input_dim=900, kernel_initializer='normal', activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(num_categories, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='categorical_crossentropy', 
              optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])
# Implement dropout for proper training set

model.fit(feature_vecs_train, train_labels, epochs=20, batch_size=2)
save_model = input('Enter file name for model to save:')
mode.save(save_model+'.h5')