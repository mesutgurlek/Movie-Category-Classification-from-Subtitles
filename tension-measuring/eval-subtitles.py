from collections import namedtuple
from itertools import groupby

def parse_subtitle(filename):
    # "chunk" our input file, delimited by blank lines
    with open(filename) as f:
        res = [list(g) for b,g in groupby(f, lambda x: bool(x.strip())) if b]

    Subtitle = namedtuple('Subtitle', 'number start end content')

    subs = []

    for sub in res:
        if len(sub) >= 3: # not strictly necessary, but better safe than sorry
            sub = [x.strip() for x in sub]
            number, start_end, *content = sub # py3 syntax
            start, end = start_end.split(' --> ')
            subs.append(Subtitle(number, start, end, content))

    return subs

subs = parse_subtitle("castaway.srt")