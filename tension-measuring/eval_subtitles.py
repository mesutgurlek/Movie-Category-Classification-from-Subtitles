from collections import namedtuple
from itertools import groupby


def parse_subtitle(filename):
    with open(filename, "rb") as f:
        res = [list(g) for b, g in groupby(f, lambda x: bool(x.strip())) if b]

    Subtitle = namedtuple('Subtitle', 'number start end content')

    subs = []

    for sub in res:
        if len(sub) >= 3:  # not strictly necessary, but better safe than sorry
            sub = [x.strip() for x in sub]
            number = sub[0].decode("UTF-8")
            start_end = sub[1].decode("UTF-8")
            content = sub[2]

            if len(start_end.split(' --> ')) == 2:
                start, end = start_end.split(' --> ')
                subs.append(Subtitle(number, start, end, content))

    return subs


subs = parse_subtitle("castaway.srt")
