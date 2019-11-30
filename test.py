import numpy as np
import pandas as pd
import re
import string

import tensorflow

# Function to to do some basic cleanup
def clean_all(text):
    text = text.lower()  # make lowercase
    text = text.encode('ascii', 'ignore').decode('ascii')  # remove emoji characters
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuation
    return text


# Load Glove model, publicly available
file = open("glove.6B.300d.txt", encoding="utf8")

# Save in dictionary
word_vecs = {}
for line in file:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], "float32")
    word_vecs[word] = vector

filename = input('Enter csv filename for testing:')
df = pd.read_csv(filename + '.csv')
model_name = input('Enter model name:')
model = tensorflow.keras.models.load_model(model_name + '.h5')

num_features = df.shape[0]
num_categories = 3
words_dim = 300

df['cleaned_feat'] = df['features'].apply(clean_all)
feature_vecs = []
for feature in df.cleaned_feat:
    lst = feature.split()
    num_words = len(lst)
    feature_vec = np.zeros((1, words_dim))
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
                x = word_vecs[word2]
                x = np.expand_dims(np.array(x), axis=-1)
                feature_vec = np.concatenate((feature_vec, x.T))
    feature_vec = feature_vec[1:, :]
    feature_vecs.append(feature_vec)
feature_vecs = np.array(feature_vecs)
feature_vecs_test = np.zeros((num_features, num_categories, words_dim))

for vec_n in range(len(feature_vecs)):
    vec = feature_vecs[vec_n]
    u, s, vh = np.linalg.svd(vec, full_matrices=True)
    vec_reduced = vh[0:3, :]
    feature_vecs_test[vec_n, :, :] = vec_reduced
feature_vecs_test = np.reshape(feature_vecs_test, newshape=(df.shape[0], -1))
out = model.predict(feature_vecs_test)
df['tech_debt'] = out[:, 0]
df['security'] = out[:, 1]
df['arch'] = out[:, 2]
df = df.drop(['cleaned_feat'], axis=1)

output_filename = input('Enter file name for output to save:')
df.to_csv(output_filename + '.csv')