import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from os import path
import numpy as np
from tokenization import *

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import global_variables

categories = global_variables.genres
def text2num(text):

    nums = range(len(categories))
    num_ver = []
    for label in text:
        num_ver.append([nums[categories.index(label,0,len(categories))]])

    return num_ver







# Categorize words and plot them
category_dict = categorize_words(path.relpath("ProcessedNormalText"))
to_be_filtered = ['im', 'oh', 'dont', 'go', 'know', 'yeah', 'come', 'get', 'well']  # 'grunt', 'beep', 'grunts', ',', 'groan', 'speak', 'music']

test_size = 500

# ProcessedNormalText has the whole data
# CategoryData has the hearing impared data
text, genre = tag_subtitles(path.relpath('ProcessedNormalText'))


for i in range(len(text)):
    for f in to_be_filtered:
        text[i] = text[i].replace(f, '')



#clf = MultinomialNB(alpha=a) #naive bayes
#clf = LogisticRegression(C=a, max_iter= 1000)
#clf = neighbors.KNeighborsClassifier( a * 10, 'distance') # knn
clf = svm.LinearSVC() #support vector machine

acc = 0
sample_num = 1
for i in range(sample_num):
    text, genre = randomize(text, genre)

    X_train = np.array(text)
    y_train_text1 = text2num(genre)

    X_test = text[0:test_size]

    y_train_text = MultiLabelBinarizer().fit_transform(y_train_text1)

    target_names = categories

    lb = preprocessing.LabelBinarizer()
    Y = lb.fit_transform(y_train_text)

    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(clf))])

    classifier.fit(X_train, Y)
    predicted = classifier.predict(X_test)
    all_labels = lb.inverse_transform(predicted)

    cor_guess = 0
    for idx,label in enumerate(all_labels):
        #print(label)
        #print(y_train_text1[idx])

        if label[y_train_text1[idx]]:
            cor_guess = cor_guess + 1

    print("accuracy = " + str( cor_guess / len(all_labels) * 100.0))
