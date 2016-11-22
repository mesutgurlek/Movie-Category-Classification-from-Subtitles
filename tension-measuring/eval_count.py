import os
from eval_subtitles import parse_subtitle
import numpy as np
import datetime as dt
from sklearn.neighbors import KNeighborsClassifier
from datetime import timedelta
import matplotlib.pyplot as plt

# BELOW INITS ARE FOR SINGLE SUBTITLE PARSING ONLY, NOT FOR KNN IMPLEMENTATION
filename = "/home/burak/Documents/Courses-2016f/CS464/Project/Subtitles/Romance/Twilight (IMPAIRED).srt"
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
        for filename in sub_files[:-15]:
            subs = parse_subtitle(os.path.join(dirpath, filename))
            if len(subs) <= 0:
                continue

            training_dataset.append(count_percentage(subs))
            training_labels.append(category_name)

        for filename in sub_files[-15:]:
            subs = parse_subtitle(os.path.join(dirpath, filename))
            if len(subs) <= 0:
                continue

            cv_dataset.append(count_percentage(subs))
            cv_labels.append(category_name)


    result = { 'training': (training_dataset, training_labels),
               'cv': (cv_dataset, cv_labels)}

    return result

# KNN TRAINING AND TESTING STEP
def knn_train_and_test():
    dataset = knn_init_dataset_cvset()

    # for k in range(2,13):
    for k in range(2,3):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(np.array(dataset['training'][0], dtype='int_'), dataset['training'][1])
        # neigh.predict_proba(dataset['validation'][0])
        score = neigh.score(dataset['validation'][0], dataset['validation'][1])
        print("Score of scikitlearn on this with K=%d ==> %f" % (k, score))

    # def sklearn_accuracy():
#     # X = [[0], [1], [2], [3]]
#     dataset = read_csv_to_matrix('')
#     training = dataset['training']
#     validation = dataset['validation']
#     test = dataset['test']
#
#     trainingX = training['x']
#     y = training['y']
#     trainingY = np.array(y.T)[0]
#
#
#
#     #cross-validation data
#     accuracies = []
#     for k in [1] + list(range(2, 17, 2)):
#     # for k in range(2, 13):
#         neigh = KNeighborsClassifier(n_neighbors=k)
#         neigh.fit(trainingX, trainingY)
#         neigh.predict_proba(validation['x'])
#         score = neigh.score(validation['x'], validation['y'])
#         print("Score of scikitlearn on this with K=%d ==> %f" % (k, score))
#
#     pprint.pprint(accuracies)
#


#gives the percentage dialog count array
def count_percentage(subs):
    end_of_movie = subs[-1].end
    movie_length_mins = int(str_to_timedelta(end_of_movie).total_seconds() / 60 ) + 1
    second_interval = (movie_length_mins + 1) * 60 / 100.0
    counts = [0] * 100 #this will give the percentage dialog count array

    secs = second_interval
    index = 0
    for sub in subs:
        if sub.at_seconds < secs:
            counts[index] += 1
        else:
            secs += second_interval
            index += 1
            counts[index] += 1

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
# plot_counts_perc(counts_perc)

knn_train_and_test()