import pickle
from sklearn import svm
from os import path
from os import listdir
from tension_measuring.eval_subtitles import *
import os.path
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle
from preprocess import global_variables

min_wpm = 999
max_wpm = 0
dump_filename = 'wpm_new.pickle'

dataset = []


class KnnDpmWpm:
    def read_files(self, trainPath, testPath):
        for dir in listdir(trainPath):
            # print(dir)
            for file in listdir(trainPath + "/" + dir):
                # print(dir + "/" + file)
                for f in self.dataset['training']:
                    if f['filename'] == file:
                        f['labels'] = dir
                        self.training_data.append(f)
                        # print(f)
                        break
        for dir in listdir(testPath):
            for file in listdir(testPath + "/" + dir):

                flag = True
                for f in self.dataset['training']:
                    if f['filename'] == file:
                        f['labels'] = dir
                        self.test_data.append(f)
                        # print(f)
                        flag = False
                        break
                if flag:
                    print('bulamiyoom', file)

    def train_model(self):
        best_accuracy = 0
        self.best_k = 0
        for k in range(1, 2):
            values, labels = shuffle([f['values'] for f in self.training_data],
                                     [f['labels'] for f in self.training_data])

            training_values = values[:(int)(4 * len(values) / 5)]
            training_labels = labels[:(int)(4 * len(labels) / 5)]

            test_values = values[(int)(4 * len(values) / 5 + 1):]
            test_labels = labels[(int)(4 * len(labels) / 5 + 1):]

            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(np.array(training_values, dtype='float_'), training_labels)

            accuracy = self.test_model(model, test_values, test_labels)
            print(k, accuracy)
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                self.best_k = k
        print(self.best_k, best_accuracy)
        self.model = KNeighborsClassifier(n_neighbors=self.best_k)
        self.model.fit(np.array([f['values'] for f in self.training_data], dtype='float_'),
                       [f['labels'] for f in self.training_data])

    def test_model(self, model, test_values, test_labels):
        count_true = 0
        for i in range(len(test_values)):
            predicted = model.predict([test_values[i]])[0]
            if predicted == test_labels[i]:
                count_true += 1
        # print("accuracy: " + str((100.0 * count_true) / len(test_values)))
        return (100.0 * count_true) / len(test_values)

    def predict(self, file_name):
        # print("filename(old): ", file_name)
        file_name = file_name.split("/")[-1]
        # print("filename(new): ", file_name)
        for f in self.test_data:
            # if f['filename'] == 'Alex Cross (IMPAIRED).srt':
            #     print('buldum')
            # else:
            #     print('bulamadim')
            #if file_name == 'Broken Arrow (IMPAIRED).srt':
            #    print(f['filename'])
            if f['filename'] == file_name:
                return self.model.predict([f['values']])[0]

    def get_f1_scores(self):
        values, labels = shuffle([f['values'] for f in self.training_data],
                                 [f['labels'] for f in self.training_data])
        training_values = values[:(int)(4 * len(values) / 5)]
        training_labels = labels[:(int)(4 * len(labels) / 5)]

        test_values = values[(int)(4 * len(values) / 5 + 1):]
        test_labels = labels[(int)(4 * len(labels) / 5 + 1):]

        model = KNeighborsClassifier(n_neighbors=self.best_k)
        model.fit(np.array(training_values, dtype='float_'), training_labels)

        #self.model = KNeighborsClassifier(n_neighbors=self.best_k)
        #model.fit(np.array(training_values, dtype='float_'), training_labels)

        predictions = []
        true_labels = []
        count_true = 0
        for i in range(len(test_values)):
            predicted = model.predict([test_values[i]])[0]
            if predicted == test_labels[i]:
                count_true += 1
            predictions.append(predicted)
            true_labels.append(test_labels[i])

        # for test in self.test_data:
        #     predicted = self.model.predict([test['values']])[0]
        #     if predicted == test['labels']:
        #         count_true += 1
        #     predictions.append(predicted)
        #     true_labels.append(test['labels'])

        print("accuracy: " + str((100.0 * count_true) / len(self.test_data)))
        print(classification_report(true_labels, predictions))
        p, r, f1, s = precision_recall_fscore_support(true_labels, predictions)
        f1_scores = [float("{0:.2f}".format(a)) for a in f1]
        #print(f1_scores)
        f1_dict = {}
        for idx, cat in enumerate(global_variables.genres):
            f1_dict[cat] = f1_scores[idx]
        #print(f1_dict)
        return f1_dict

    def load_dataset(self):
        f = open(self.pickleFile, 'rb')
        dataset = pickle.load(f)
        f.close()
        return dataset

    def __init__(self, train_path, test_path):
        self.clf = None
        self.optimal_alpha = None
        self.train_path = path.relpath(train_path)
        self.test_path = path.relpath(test_path)
        self.model = None
        self.best_k = 0
        self.dataset = None
        self.pickleFile = '../tension_measuring/wpm_new.pickle'
        self.training_data = []
        self.test_data = []

        if os.path.isfile(self.pickleFile):
            self.dataset = self.load_dataset()
        else:
            print("pickle yok")
        print(self.train_path)
        print(self.test_path)
        self.read_files("../" + self.train_path, "../" + self.test_path)

        self.train_model()
        # self.test_model()


def main():
    global dataset

    if not os.path.isfile(dump_filename):
        dataset = generate_dataset_wpm_dpm()
        dump_dataset(dataset)
    else:
        dataset = load_dataset()

        # test()


def generate_dataset_wpm_dpm():
    global dataset
    dataset = {'training': [],
               'testing': []}

    count_movie = 0

    # for dirpath, dirnames, filenames in os.walk("../NonImpairedSubtitles"):
    for dirpath, dirnames, filenames in os.walk("../Subtitles"):
        dirname = dirpath.split("/")[-1]

        if dirname == "Training" or dirname == "Adventure":
            continue
        print(dirname)

        cnt = 0
        indices = []

        files = [f for f in filenames if f.endswith(".srt")]
        for index, filename in enumerate(files):
            # if filename == 'Broken Arrow (IMPAIRED).srt':
            #     print(os.path.join(dirpath, filename))
            subs = parse_subtitle(os.path.join(dirpath, filename))
            if len(subs) <= 0:
                continue
            word_count = 0
            for sub in subs:
                word_count += len(str(sub.content).split(" "))

            cnt += 1
            count_movie += 1
            movie_time_minute = 60 * int(subs[-1].start.split(":")[0]) + int(subs[-2].start.split(":")[1])
            if movie_time_minute <= 0:
                continue

            word_per_minute = word_count / movie_time_minute

            global min_wpm
            global max_wpm

            if min_wpm > word_per_minute:
                min_wpm = word_per_minute
            if max_wpm < word_per_minute:
                max_wpm = word_per_minute

            indices.append(count_movie)

            dialog_per_minute = len(subs) / movie_time_minute
            indices.append(count_movie)

            movie_time_minute = 60 * int(subs[-1].start.split(":")[0]) + int(subs[-1].start.split(":")[1])
            if movie_time_minute <= 0:
                continue

            # if index < len(files) - 200:
            tmp = {'values': [dialog_per_minute, word_per_minute], 'labels': dirname, 'filename': filename,
                   'movie_times_minute': movie_time_minute}
            dataset['training'].append(tmp)
            # dataset['training']['values'].append([dialog_per_minute, word_per_minute])
            # dataset['training']['labels'].append(dirname)
            # dataset['training']['filenames'].append(filename)
            # dataset['training']['movie_times_minute'].append(movie_time_minute)
            # else:
            #    tmp = {'values': [dialog_per_minute, word_per_minute], 'labels': dirname, 'filename': filename,
            #           'movie_times_minute': movie_time_minute}
            #    dataset['testing'].append(tmp)

            # dataset['testing']['values'].append([dialog_per_minute, word_per_minute])
            # dataset['testing']['labels'].append(dirname)
            # dataset['testing']['filenames'].append(filename)
            # dataset['testing']['movie_times_minute'].append(movie_time_minute)

    return dataset


def dump_dataset(dataset):
    f = open(dump_filename, 'wb')
    pickle.dump(dataset, f)
    f.close()


def load_dataset():
    f = open(dump_filename, 'rb')
    dataset = pickle.load(f)
    f.close()
    return dataset


def makeTest(k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(np.array(dataset['training']['values'], dtype='float_'), dataset['training']['labels'])

    count_true = 0
    count_movie = len(dataset['testing']['values'])

    test_labels = []
    predictions = []

    for i in range(count_movie):
        predicted = neigh.predict([dataset['testing']['values'][i]])[0]
        true_label = dataset['testing']['labels'][i]

        # if i < 10:
        #     print(true_label, predicted)
        #     print(dataset['testing']['values'][i], dataset['testing']['labels'][i], dataset['testing']['filenames'][i])
        test_labels.append(true_label)
        predictions.append(predicted)

        if true_label == predicted:
            count_true += 1

    print(classification_report(test_labels, predictions))
    return 100. * count_true / count_movie


def test():
    k_values = []
    accuracies = []
    for k in range(110, 111):
        accuracy = makeTest(k)
        k_values.append(k)
        accuracies.append(accuracy)
        print(k, accuracy)

    plt.figure(figsize=(12, 9), dpi=80)

    plt.plot(k_values, accuracies, 'o')
    plt.axis([1, 250, -1, 61])

    plt.xlabel('K values')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right', numpoints=1)
    plt.title("Accuracies / K Values")

    for k, accuracy in zip(k_values, accuracies):
        plt.text(k - 0.6, accuracy + 1, str(k) + ", " + str(format(accuracy, '.1f')), fontsize=10)

        # plt.show()

#main()
