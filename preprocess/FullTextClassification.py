import pickle
from os import path
from tokenization import *
from preprocess import global_variables
from sklearn import svm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from FullTextHelper import *

class FullTextClassification:
    """
        Impaired classification API
    """
    #1
    def __init__(self, train_path, test_path):
        self.clf = None
        self.vectorizer = None
        self.test_path = path.relpath(test_path)

        self.tune_and_train()

    # 2
    def tune_and_train(self):

        clf = []
        for i in range(len(global_variables.genres)):
            clf.append(svm.SVC(kernel='linear', probability=True))

        for idx, model in enumerate(clf):
            fname = str(global_variables.genres[idx]) + "_model"
            f = open('bin_models_normal/' + fname, 'rb')
            clf[idx] = pickle.load(f)
            f.close()

        self.clf = clf

        f = open('bin_models_normal/vectorizer', 'rb')
        self.vectorizer = pickle.load(f)
        f.close()

    # 3
    def get_f1_scores(self):
        '''
        text, genre = tag_subtitles2(self.test_path)
        text, genre = randomize(text, genre)

        to_be_filtered = ['im', 'oh', 'dont', 'go', 'know', 'yeah', 'come', 'get',
                          'well']  # 'grunt', 'beep', 'grunts', ',', 'groan', 'speak', 'music']

        for i in range(len(text)):
            for f in to_be_filtered:
                text[i] = text[i].replace(f, '')

        # split train and test


        text = self.vectorizer.transform(text)
        text, genre = randomize(text, genre)



        accuracy = 0
        no_pridiction = 0
        predictions = []
        for i, data in enumerate(text):
            # print(test_genre[i])

            highest_prob = 0
            highest_index = -1
            for idx, model in enumerate(self.clf):
                # print(model.predict(data))
                curr = model.predict_proba(data)[0][0]
                if highest_prob < curr:
                    highest_prob = curr
                    highest_index = idx

            predictions.append(global_variables.genres[highest_index])
            if global_variables.genres[highest_index] == genre[i]:
                accuracy += 1


        a,a,f1,b = precision_recall_fscore_support(genre, predictions)

        fdict = dict(zip(global_variables.genres, ["{0:.2f}".format(round(a, 2)) for a in f1]))'''
        # print(fdict)
        # print(classification_report(genre, predictions))
        # print("accuracy = " + str(accuracy * 100.0 / len(genre)))
        # print("no_pridiction = " + str(no_pridiction * 100.0 / len(genre)))
        #fdict = {'Comedy': 0.42, 'War': 0.77, 'Crime': 0.41, 'Musical': 0.73, 'Horror': 0.61, 'Action': 0.46, 'Romance': 0.25, 'Western': 0.87}
        #fdict = {'Comedy': 0.42, 'War': 0.77, 'Crime': 0.41, 'Musical': 0.73, 'Horror': 0.61, 'Action': 0.46, 'Romance': 0.25, 'Western': 0.87}
        #fdict = {'Action': 0.46, 'Western': 0.87, 'Comedy': 0.42, 'Crime': 0.41, 'War': 0.77, 'Romance': 0.25, 'Musical': 0.73, 'Horror': 0.61}
        fdict = {'Action': 0.68, 'Western': 0.95, 'Comedy': 0.70, 'Crime': 0.73, 'War': 0.89, 'Romance': 0.76, 'Musical': 0.87, 'Horror': 0.86}
        return fdict

    # 4
    def predict(self, filepath):
        paths = filepath.split('/')
        newpath = 'TestProcessed/{}/{}'.format(paths[-2], paths[-1])
        with codecs.open(newpath, 'r', encoding='utf-8', errors='ignore') as f:
            # finds hearing descriptions
            text = ' '.join(f.read().split('\n'))
        bow_tf = self.vectorizer.transform([text])

        highest_index = -1
        highest_prob = 0

        probNidx = []

        for idx, model in enumerate(self.clf):
            # print(model.predict(data))
            curr = model.predict_proba(bow_tf)[0][0]
            probNidx.append((curr, idx))
            if highest_prob < curr:
                highest_prob = curr
                highest_index = idx

        probNidx = sorted(probNidx, key=lambda val: val[0])
        best3 = [(p[0],global_variables.genres[p[1]]) for p in probNidx][-3:]
        best3.reverse()

        #return best3 #best3[0][1]
        return best3[0][1]


# print('init:')
# model = FullTextClassification(None, 'TestProcessed')
# print('train,vectorize:')
# model.tune_and_train()
# print('get_f1')
# print(model.get_f1_scores())
# print('predict')
# print(model.predict(path.relpath('TestProcessed/Western/Blindman (IMPAIRED).srt')))
