import json
import os
from eval_subtitles import parse_subtitle
import numpy as np
import datetime as dt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from datetime import timedelta
import matplotlib.pyplot as plt

dataset_cache_path = "./sub_count_set_cache.txt"

# BELOW INITS ARE FOR SINGLE SUBTITLE PARSING ONLY, NOT FOR KNN IMPLEMENTATION
filename = "/home/burak/Documents/Courses-2016f/CS464/Project/Subtitles/Romance/Cast Away (IMPAIRED).srt"
subs = parse_subtitle(filename)
movie_name = filename.split('/')[-1]


def str_to_timedelta(str_tm="02:14:53,085"):
    hours, mins, secs = str_tm.split(':')
    millis = secs[3:]
    secs = secs[:2]
    return timedelta(hours=int(hours), minutes=int(mins), seconds=int(secs), milliseconds=int(millis))

def knn_init_dataset_cvset():
    training_dataset = []
    training_labels = []
    cv_dataset = []
    cv_labels = []

    # INITIALIZE THE DATA SET
    for dirpath, dirnames, filenames in os.walk("../Subtitles"):
        category_name = dirpath.split("/")[-1]

        if category_name == "Subtitles":
            continue
        print(category_name)

        sub_files = [f for f in filenames if f.endswith(".srt")]
        for filename in sub_files[:-45]:
            subs = parse_subtitle(os.path.join(dirpath, filename))
            if len(subs) <= 0:
                continue

            counts = count_percentage(subs)
            if counts is not None:
                training_dataset.append(counts)
                training_labels.append(category_name)

        for filename in sub_files[-45:]:
            subs = parse_subtitle(os.path.join(dirpath, filename))
            if len(subs) <= 0:
                continue

            counts = count_percentage(subs)
            if counts is not None:
                cv_dataset.append(counts)
                cv_labels.append(category_name)


    result = { 'training': (training_dataset, training_labels),
               'cv': (cv_dataset, cv_labels)}

    normalize_count_feature(training_dataset)
    normalize_count_feature(cv_dataset)

    try:
        with open(dataset_cache_path, mode='w') as file:
            file.write(json.dumps(result))
            print("Cache file generated at: %s" % dataset_cache_path)
    except Exception:
        print("Couldn't store the dataset in a cache file in path:%s" % dataset_cache_path)

    return result


def normalize_count_feature(dataset):
    mins_of_counts = [min(data) for data in dataset]
    maxs_of_counts = [max(data) for data in dataset]
    final_min, final_max = min(mins_of_counts), max(maxs_of_counts)

    for row in range(len(dataset)):
        for col in range(len(dataset[row])):
            dataset[row][col] = (dataset[row][col] - final_min) / (final_max - final_min)


# KNN TRAINING AND TESTING STEP
def knn_train_and_test():
    try: #try loading the cache file
        with open(dataset_cache_path, mode='r') as file:
            dataset = json.load(file)
    except Exception:
        dataset = knn_init_dataset_cvset()

    for k in range(2,13):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(dataset['training'][0], dataset['training'][1])
        # neigh.predict_proba(dataset['cv'][0])
        score = neigh.score(dataset['cv'][0], dataset['cv'][1])
        print("Score of scikitlearn on this with K=%d ==> %f" % (k, score))

    # clf = MultinomialNB(alpha=0.01) #naive bayes
    # clf = LogisticRegression(C=0.01, max_iter= 1000)
    # clf = svm.SVC()
    # clf.fit(dataset['training'][0], dataset['training'][1])
    # score = clf.score(dataset['cv'][0], dataset['cv'][1])
    # print("Score ==> %f" % (score))


#gives the percentage dialog count array
def count_percentage(subs):
    if len(subs) <= 100:
        return None #this subtitle is useless, throw it away

    if str_to_timedelta(subs[-20].start) > str_to_timedelta(subs[-1].end):
        # this check is for fixing some buggy subtitles which has wrong end time label
        end_of_movie = subs[-3].end
    else:
        end_of_movie = subs[-1].end
    movie_length_mins = int(str_to_timedelta(end_of_movie).total_seconds() / 60 ) + 2
    second_interval = (movie_length_mins + 1) * 60 / 100.0
    counts = [0] * 100 #this will give the percentage dialog count array

    secs = second_interval
    index = 0
    try:
        for sub in subs:
            if sub.at_seconds < secs:
                counts[index] += 1
            else:
                secs += second_interval
                index += 1
                counts[index] += 1
    except IndexError:
        pass

    return counts if len(counts) == 100 else None


def count_intervals(subs, minute_interval):
        end_of_movie = subs[-1].end
        movie_length_mins = int(str_to_timedelta(end_of_movie).total_seconds() / 60 ) + 1
        counts = [0] * (int(movie_length_mins / minute_interval) + 1)

        mins = minute_interval
        index = 0
        for sub in subs:
            if sub.at_minute < mins:
                counts[index] += 1
            else:
                mins += minute_interval
                index += 1
                counts[index] += 1

        return counts


def plot_counts(counts, interval):

    plt.figure(figsize=(12, 9), dpi=80)

    plt.plot(range(0, len(counts)*interval, interval), counts, label='Counts')
    plt.interactive(False)

    plt.xlabel('Minutes')
    plt.ylabel('Dialog Count')
    # plt.legend(loc='upper right', numpoints = 1)
    plt.title("Dialog Tension of %s" % movie_name)
    # plt.show(block=True)
    plt.savefig('%s.png' % movie_name)
    print(counts)


def plot_counts_percentage(counts):

    plt.figure(figsize=(12, 9), dpi=80)

    plt.plot(range(1, 101), counts, label='Counts')
    plt.interactive(False)

    plt.xlabel('Percentage')
    plt.ylabel('Dialog Count')
    # plt.legend(loc='upper right', numpoints = 1)
    plt.title("Dialog Tension of %s" % movie_name)
    # plt.show(block=True)
    plt.savefig('PERC_%s.png' % movie_name)
    print(counts)


# interval = 1
# counts = count_intervals(subs, interval)
# plot_counts(counts, interval)
# counts_perc = count_percentage(subs)
# plot_counts_percentage(counts_perc)

knn_train_and_test()