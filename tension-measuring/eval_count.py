from eval_subtitles import parse_subtitle
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt

subs = parse_subtitle("/home/burak/Documents/Courses-2016f/CS464/Project/Subtitles/Action/The Dark Knight Rises (IMPAIRED)")



def str_to_timedelta(str_tm="02:14:53,085"):
    hours, mins, secs = str_tm.split(':')
    millis = secs[3:]
    secs = secs[:2]
    return timedelta(hours=int(hours), minutes=int(mins), seconds=int(secs), milliseconds=int(millis))


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


counts = count_intervals(subs, 5)

plt.figure(figsize=(12, 9), dpi=80)

# def plot_counts(counts):
plt.plot(range(len(counts)), counts, label='Counts')
# print(indices, " ", dialog_per_minutes)
# if cnt > 0:
#     print(sum/cnt)
plt.interactive(False)

plt.xlabel('Movies')
plt.ylabel('Dialog per Minute')
plt.legend(loc='upper right', numpoints = 1)
plt.title("Dialog per Minute / Movie Genres")
plt.show(block=True)
plt.savefig('foo.png')
print(counts)
