from eval_subtitles import *
import os.path
import matplotlib.pyplot as plt

count_movie = 0
plt.figure(figsize=(12, 9), dpi=80)
for dirpath, dirnames, filenames in os.walk("../NonImpairedSubtitles"):
    dirname = dirpath.split("/")[-1]

    if dirname == "NonImpairedSubtitles":
        continue
    #print(dirname)
    cnt = 0
    sum = 0
    indices = []
    dialog_per_minutes = []
    for filename in [f for f in filenames if f.endswith(".srt")]:
        subs = parse_subtitle(os.path.join(dirpath, filename))
        if len(subs) <= 0:
            continue
        cnt += 1
        count_movie += 1
        movie_time_minute = 60*int(subs[-1].start.split(":")[0]) + int(subs[-1].start.split(":")[1])
        if movie_time_minute <= 0:
            continue
        dialog_per_minute = len(subs)/movie_time_minute
        indices.append(count_movie)
        dialog_per_minutes.append(dialog_per_minute)
        sum += dialog_per_minute
    plt.plot(indices, dialog_per_minutes,'o', label=dirname)
    #print(indices, " ", dialog_per_minutes)
    plt.axis([0, count_movie+10, 0, 40])
    if cnt > 0:
        print(dirname, sum/cnt)

plt.xlabel('Movies')
plt.ylabel('Dialog per Minute')
plt.legend(loc='upper right', numpoints = 1)
plt.title("Dialog per Minute / Movie Genres")
plt.show()
