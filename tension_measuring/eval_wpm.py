from eval_subtitles import *
import os.path
import matplotlib.pyplot as plt

count_movie = 0
min_wpm = 999
max_wpm = 0
plt.figure(figsize=(12, 9), dpi=80)
for dirpath, dirnames, filenames in os.walk("../Subtitles"):
    dirname = dirpath.split("/")[-1]

    if dirname == "Subtitles":
        continue
    #print(dirname)
    cnt = 0;
    sum = 0;
    indices = []
    word_per_minutes = []
    for filename in [f for f in filenames if f.endswith(".srt")]:
        subs = parse_subtitle(os.path.join(dirpath, filename))
        if len(subs) <= 0:
            continue
        word_count = 0
        for sub in subs:
            word_count += len(str(sub.content).split(" "))

        cnt += 1
        count_movie += 1
        movie_time_minute = 60*int(subs[-1].start.split(":")[0]) + int(subs[-1].start.split(":")[1])
        if movie_time_minute <= 0:
            continue
        word_per_minute = word_count/movie_time_minute

        if min_wpm > word_per_minute:
            min_wpm = word_per_minute
        if max_wpm < word_per_minute:
            max_wpm = word_per_minute

        indices.append(count_movie)
        word_per_minutes.append(word_per_minute)
        sum += word_per_minute
    plt.plot(indices, word_per_minutes,'o', label=dirname)
    #print(indices, " ", word_per_minutes)

    if cnt > 0:
        print(dirname, sum/cnt)

plt.axis([0, count_movie+10, 0, 150])

plt.xlabel('Movies')
plt.ylabel('Word per Minute')
plt.legend(loc='upper right', numpoints = 1)
plt.title("Word per Minute / Movie Genres")
plt.show()