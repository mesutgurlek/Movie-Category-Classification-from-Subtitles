from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from os import path
import numpy as np
from tokenization import *
from preprocess import global_variables
import pickle
import os
from sklearn.metrics import classification_report

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


def genre2bin(genre_list, genre):
    bin_list = []
    for idx, g in enumerate(genre_list):
        if g == genre:
            bin_list.append(g)
        else:
            bin_list.append('o')

    return bin_list


#  # Categorize words and plot them
category_dict = categorize_words(path.relpath("ProcessedNormalText"))
to_be_filtered = ['im', 'oh', 'dont', 'go', 'know', 'yeah', 'come', 'get', 'well']  # 'grunt', 'beep', 'grunts', ',', 'groan', 'speak', 'music']

# for i in categories:
#     for f in to_be_filtered:
#         category_dict[i] = category_dict[i].replace(f, '')
#
# for c in categories:
#     cleaned_list = clean_stopword(category_dict[c])
#     stemmed_data = stemming(cleaned_list)
#
#     fd = nltk.FreqDist(stemmed_data)
#     print('Category: ', c)
#     print(fd.most_common(12))
#     fd.plot(12, cumulative=False)

# process_movie_subtitles(path.relpath("ProcessedSubtitles"), path.relpath("CategoryData"))w


test_size = 1000

# ProcessedNormalText has the whole data
# CategoryData has the hearing impared data
text, genre = tag_subtitles(path.relpath('ProcessedNormalText'))


for i in range(len(text)):
    for f in to_be_filtered:
        text[i] = text[i].replace(f, '')

sample_num = len(global_variables.genres)
print(sample_num)

#clf = MultinomialNB(alpha=a) #naive bayes
#clf = LogisticRegression(C=a, max_iter= 1000)
#clf = neighbors.KNeighborsClassifier( a * 10, 'distance') # knn
clf = []#support vector machine

for i in range(sample_num):
    clf.append(svm.SVC(kernel='linear', probability=True))

text, genre = randomize(text, genre)

'''
for i in range(sample_num):
    text, genre = randomize(text, genre)

    bin_genre = genre2bin(genre, global_variables.genres[i])

    bow_tf = bag_of_words_and_tf(text)

    clf[i].fit(bow_tf[test_size:], bin_genre[test_size:])

    test_data = bow_tf[:test_size]
    test_genre_bin = bin_genre[:test_size]
    print('.')
    predicted = clf[i].predict(test_data)

    print(str(accuracy_score(test_genre_bin, predicted)*100))
'''
text, genre = randomize(text, genre)

#get accuracy
test_genre = genre[:test_size]
test_data = bag_of_words_and_tf(text)[:test_size]
test_data, test_genre = randomize(test_data, test_genre)

'''
#save
for idx, model in enumerate(clf):
    fname = str(global_variables.genres[idx]) + "_model"
    f = open('bin_models_normal/' + fname, 'wb')
    pickle.dump(model, f)
    f.close()

'''
# load
for idx, model in enumerate(clf):
    fname = str(global_variables.genres[idx]) + "_model"
    print(fname)
    f = open('bin_models_normal/' + fname, 'rb')
    clf[idx] = pickle.load(f)
    f.close()


accuracy = 0
no_pridiction = 0
predictions = []
for i,data in enumerate(test_data):
    #print(test_genre[i])

    highest_prob = 0
    highest_index = -1
    for idx,model in enumerate(clf):
        #print(model.predict(data))
        curr = model.predict_proba(data)[0][0]
        if highest_prob < curr:
            highest_prob = curr
            highest_index = idx

    predictions.append(global_variables.genres[highest_index])
    if global_variables.genres[highest_index] == test_genre[i]:
        accuracy += 1

print(classification_report(test_genre, predictions))
print("accuracy = " + str(accuracy * 100.0 / len(test_genre)))
print("no_pridiction = " + str(no_pridiction * 100.0 / len(test_genre)))



