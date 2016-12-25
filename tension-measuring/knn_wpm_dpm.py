import pickle
from sklearn import svm
from os import path
from eval_subtitles import *
import os.path
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report

min_wpm = 999
max_wpm = 0
dump_filename = 'wpm_dpm_calc.ali'

dataset = {}


def main():
    global dataset

    if not os.path.isfile(dump_filename):
        dataset = generate_dataset_wpm_dpm()
        dump_dataset(dataset)
    else:
        dataset = load_dataset()

    test()


def generate_dataset_wpm_dpm():
    dataset = {'training': {
        'values': [],
        'labels': [],
        'filenames': [],
        'movie_times_minute': []},
        'testing': {
            'values': [],
            'labels': [],
            'filenames': [],
            'movie_times_minute': []}}

    count_movie = 0

    for dirpath, dirnames, filenames in os.walk("../NonImpairedSubtitles"):
        dirname = dirpath.split("/")[-1]

        if dirname == "Training" or dirname == "Adventure":
            continue
        print(dirname)

        cnt = 0
        indices = []

        files = [f for f in filenames if f.endswith(".srt")]
        for index, filename in enumerate(files):
            subs = parse_subtitle(os.path.join(dirpath, filename))
            if len(subs) <= 0:
                continue
            word_count = 0
            for sub in subs:
                word_count += len(str(sub.content).split(" "))

            cnt += 1
            count_movie += 1
            movie_time_minute = 60 * int(subs[-1].start.split(":")[0]) + int(subs[-1].start.split(":")[1])
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

            if index < len(files) - 200:
                dataset['training']['values'].append([dialog_per_minute, word_per_minute])
                dataset['training']['labels'].append(dirname)
                dataset['training']['filenames'].append(filename)
                dataset['training']['movie_times_minute'].append(movie_time_minute)
            else:
                dataset['testing']['values'].append([dialog_per_minute, word_per_minute])
                dataset['testing']['labels'].append(dirname)
                dataset['testing']['filenames'].append(filename)
                dataset['testing']['movie_times_minute'].append(movie_time_minute)

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
    for k in range(1, 30):
        accuracy = makeTest(k)
        k_values.append(k)
        accuracies.append(accuracy)
        print(k, accuracy)

    plt.figure(figsize=(12, 9), dpi=80)

    plt.plot(k_values, accuracies, 'o')
    plt.axis([0, 31, -1, 41])

    plt.xlabel('K values')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right', numpoints=1)
    plt.title("Accuracies / K Values")

    for k, accuracy in zip(k_values, accuracies):
        plt.text(k - 0.6, accuracy + 1, str(k) + ", " + str(format(accuracy, '.1f')), fontsize=10)

        # plt.show()

main()