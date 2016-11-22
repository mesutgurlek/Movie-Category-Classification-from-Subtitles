from eval_subtitles import *
import os.path
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

count_movie = 0
min_wpm = 999
max_wpm = 0

values = []
labels = []
plt.figure(figsize=(12, 9), dpi=80)


def makeTest(k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(np.array(values, dtype='float_'), labels)

    count_movie = 0
    count_true = 0
    for dirpath, dirnames, filenames in os.walk("../Subtitles"):
        dirname = dirpath.split("/")[-1]

        if dirname == "Test":
            continue
        cnt = 0
        indices = []
        word_per_minutes = []
        dialog_per_minutes = []
        for filename in [f for f in filenames if f.endswith(".srt")][-15:]:
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

            indices.append(count_movie)
            word_per_minutes.append(word_per_minute)

            dialog_per_minute = len(subs) / movie_time_minute
            indices.append(count_movie)
            dialog_per_minutes.append(dialog_per_minute)

            predicted = neigh.predict([[dialog_per_minute, word_per_minute]])
            if dirname == predicted:
                count_true += 1
    return 100. * count_true / count_movie


for dirpath, dirnames, filenames in os.walk("../Subtitles"):
    dirname = dirpath.split("/")[-1]

    if dirname == "Training":
        continue
    print(dirname)
    cnt = 0
    sum = 0
    indices = []
    word_per_minutes = []
    dialog_per_minutes = []
    for filename in [f for f in filenames if f.endswith(".srt")][:-15]:
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

        if min_wpm > word_per_minute:
            min_wpm = word_per_minute
        if max_wpm < word_per_minute:
            max_wpm = word_per_minute

        indices.append(count_movie)
        word_per_minutes.append(word_per_minute)

        dialog_per_minute = len(subs) / movie_time_minute
        indices.append(count_movie)
        dialog_per_minutes.append(dialog_per_minute)

        values.append([dialog_per_minute, word_per_minute])
        labels.append(dirname)

k_values = []
accuracies = []
for k in range(1, 30):
    accuracy = makeTest(k)
    k_values.append(k)
    accuracies.append(accuracy)
    print(k, accuracy)

plt.plot(k_values, accuracies, 'o')
plt.axis([0, 31, -1, 41])

plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.legend(loc='upper right', numpoints=1)
plt.title("Accuracies / K Values")

for k, accuracy in zip(k_values, accuracies):
    plt.text(k - 0.6, accuracy+1, str(k) + ", " + str(format(accuracy, '.1f')), fontsize=10)

plt.show()
