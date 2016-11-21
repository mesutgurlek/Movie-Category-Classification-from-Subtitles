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

for dirpath, dirnames, filenames in os.walk("../Training"):
    dirname = dirpath.split("/")[-1]

    if dirname == "Training":
        continue
    print(dirname)
    cnt = 0
    sum = 0
    indices = []
    word_per_minutes = []
    dialog_per_minutes = []
    for filename in [f for f in filenames if f.endswith(".srt")]:
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

neigh = KNeighborsClassifier(n_neighbors=19)
neigh.fit(np.array(values, dtype='float_'), labels)


count_movie = 0
count_true = 0
for dirpath, dirnames, filenames in os.walk("../Test"):
    dirname = dirpath.split("/")[-1]

    if dirname == "Test":
        continue
    cnt = 0
    indices = []
    word_per_minutes = []
    dialog_per_minutes = []
    for filename in [f for f in filenames if f.endswith(".srt")]:
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

        predicted = neigh.predict([[dialog_per_minute, word_per_minute]])
        if dirname == predicted:
            count_true += 1
print(count_true, count_movie, 100. * count_true/ count_movie)