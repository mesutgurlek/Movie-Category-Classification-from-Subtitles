from os import path
from os import listdir
from os import mkdir
import re
import codecs
import global_variables


def filter_hearing_descriptions(line):
    res = re.findall(r'(\[.+\]|\(.+\))', line) # fix the monsters inc case.
    if res:
        res[0] = re.sub(r'<.*?>', '', res[0])
    return list(res)


def parse(file_path, output_path):
    hearing_descriptions = []
    with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        #finds hearing descriptions
        for line in f:
            hearing_descriptions.extend(filter_hearing_descriptions(line))

    # process strings
    content = '\n'.join([i[1:-1] for i in hearing_descriptions])

    if content:
        with open(output_path, 'w') as f:
            f.write(content)


def preprocess_subtitles(input_folder, output_folder):
    subtitles_path = path.relpath(input_folder)
    output_path = path.relpath(output_folder)
    categories = global_variables.genres

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

        impaired = '(IMPAIRED)'
        for f in listdir(input_folder_path):
            if impaired in f:
                # Parse hearing descriptions in subtitles
                input_subtitle = "%s/%s" % (input_folder_path, f)
                output_subtitle = "%s/%s" % (output_folder_path, f)
                parse(input_subtitle, output_subtitle)