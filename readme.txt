Download GloVe word vectors before runnning any files, then unzip:
http://nlp.stanford.edu/data/glove.840B.300d.zip


File descriptions:

- classifier_demo.ipynb: Demo of how NLP is used on the features and used by a simple neural network to generate risk probabilties.
- feature_extractor_demo.ipynb: Demo of feature extraction from design documents
- input1.csv: Training data for the NN
- test1.csv: Test data for NN
- out1.csv: Output from NN
- classification_model.h5: Saved trained classification model
- train.py: Training interface for NN
- test.py: Testing interface for NN. Run 'python test.py' which will take a trained model and an input csv file as input, and produce a csv file as output
