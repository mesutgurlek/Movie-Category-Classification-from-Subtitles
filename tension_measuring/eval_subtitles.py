from collections import namedtuple
from itertools import groupby


def parse_subtitle(filename):
    # "chunk" our input file, delimited by blank lines
    with open(filename, 'rb') as f:
        res = [list(g) for b,g in groupby(f, lambda x: bool(x.strip())) if b]

    Subtitle = namedtuple('Subtitle', 'number start end content at_minute at_seconds')

    subs = []
    number = 0
    for sub in res:
        if len(sub) >= 3: # not strictly necessary, but better safe than sorry
            sub = [x.strip() for x in sub]
            try:
                number = sub[0].decode("UTF-8")
            except:
                number += 1
            start_end = sub[1].decode("UTF-8")
            content = sub[2]
            if len(start_end.split(' --> ')) == 2:
                start, end = start_end.split(' --> ') # e.g. 02:14:53,085

                if len(start) >= 12 and len(end) >= 12:
                    start = start[:12] #for truncating unnecessary fields, if any
                    end = end[:12] #for truncating unnecessary fields, if any
                    try:
                        at_minute = int(start[:2]) * 60 + int(start[3:5])
                        at_seconds = int(start[:2]) * 3600 + int(start[3:5]) * 60 + int(start[6:8])
                    except:
                        at_minute = 0
                        at_seconds = 0
                        #continue
                    subs.append(Subtitle(number, start, end, content, at_minute, at_seconds))
            # if filename == '../Subtitles/Action/Broken Arrow (IMPAIRED).srt':
            #     print('a   ', number, start, end, content, at_minute, at_seconds)
            #     print('b   ', Subtitle(number, start, end, content, at_minute, at_seconds))
            #     print('c   ', number, start, end, content, at_minute, at_seconds)

    return subs

# subs = parse_subtitle("castaway.srt")