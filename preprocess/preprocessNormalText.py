from os import path
from os import listdir
from os import mkdir
import re
import codecs
from itertools import groupby


def parse_subtitle(filename):
    # "chunk" our input file, delimited by blank lines
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        res = [list(g) for b,g in groupby(f, lambda x: bool(x.strip())) if b]

    subs = []

    for sub in res:
        if len(sub) >= 3: # not strictly necessary, but better safe than sorry
            sub = [x.strip() for x in sub]
            content = sub[2]
            subs.append(content)

    global a
    if a == 90:
        print(filename)
        print(subs)
    return subs


def remove_impaired(content):

    regex = re.compile(r'(\[.+\]|\(.+\))')
    res = re.sub(regex, "", content)
    res = re.sub(r'<.*?>', '', res)
    return res


a = 1
def parse(file_path, output_path):

    content = parse_subtitle(file_path)

    content = " ".join([remove_impaired(mov) for mov in content])

    content = "\n".join(content.split(" "))

    if content:
        with open(output_path, 'w') as f:
            f.write(content)


def preprocess_normal_text(input_folder, output_folder):
    subtitles_path = path.relpath(input_folder)
    output_path = path.relpath(output_folder)
    categories = ['Action', 'Adventure', 'Comedy', 'Horror', 'Romance', 'War']

    for category in categories:
        input_folder_path = "%s/%s" % (subtitles_path, category)
        output_folder_path = "%s/%s" % (output_path, category)

        print('.',)
        # Create folders
        try:
            if not path.isdir(output_folder_path):
                mkdir(output_folder_path, 0o755)
        except OSError:
            print("Directorty cannot be opened in %s" % output_folder_path)

        if __name__ == '__main__':
            for f in listdir(input_folder_path):
                # Parse hearing descriptions in subtitles
                input_subtitle = "%s/%s" % (input_folder_path, f)
                output_subtitle = "%s/%s" % (output_folder_path, f)
                parse(input_subtitle, output_subtitle)


#test the code


in_path = path.relpath("Subtitles")
out_path = path.relpath("ProcessedNormalText")

preprocess_normal_text(in_path, out_path)