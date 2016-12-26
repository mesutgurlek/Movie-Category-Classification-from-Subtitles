from os import listdir
import codecs

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

def tag_subtitles2(input_folder):
    # Returns a 2d list which data[i][0] gives tag, data[i][1] gives text of i'th movie
    subtitles_path = path.relpath(input_folder)
    genres = []
    texts = []

    # get lower bound and put equal amount of catagories in the train set.
    lowest = 100000
    for category in global_variables.genres:
        input_folder_path = "%s/%s" % (subtitles_path, category)
        if len(listdir(input_folder_path)) < lowest:
            lowest = len(listdir(input_folder_path))

    print(lowest * len(global_variables.genres))

    for category in global_variables.genres:
        input_folder_path = "%s/%s" % (subtitles_path, category)
        impaired = '(IMPAIRED)'

        count = 0
        for f in listdir(input_folder_path):
            if True:
                if count > lowest:
                    break
                count += 1

                # Parse hearing descriptions in subtitles
                input_subtitle = "%s/%s" % (input_folder_path, f)
                with codecs.open(input_subtitle, 'r', encoding='utf-8', errors='ignore') as f:
                    # finds hearing descriptions
                    text = ' '.join(f.read().split('\n'))
                genres.append(category)
                texts.append(text)
    return texts, genres



def genre2bin(genre_list, genre):
    bin_list = []
    for idx, g in enumerate(genre_list):
        if g == genre:
            bin_list.append(g)
        else:
            bin_list.append('o')

    return bin_list
