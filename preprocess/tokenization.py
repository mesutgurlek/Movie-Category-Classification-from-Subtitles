from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk
from os import path
from os import mkdir
from os import listdir
import codecs

def clean_stopword(text):
    stop = set(stopwords.words('english'))
    normalized = [i for i in text.lower().split() if i not in stop]
    return normalized


def stemming(text_array):
    stemmer = PorterStemmer()
    result = [stemmer.stem(text) for text in text_array]
    return result


def plot_data(data):
    fd = nltk.FreqDist(data)
    fd.plot(30, cumulative=False)
    fd.most_common(12)


def stem_subtitles(input_folder, output_folder):
    subtitles_path = path.relpath(input_folder)
    output_path = path.relpath(output_folder)
    categories = ['Action', 'Adventure', 'Comedy', 'Horror', 'Romance', 'War']

    text_dict = {}
    for category in categories:

        text = ""
        full_text = []
        input_folder_path = "%s/%s" % (subtitles_path, category)
        output_folder_path = "%s/%s" % (output_path, category)
        # Create folders
        try:
            if not path.isdir(output_folder_path):
                mkdir(output_folder_path, 0o755)
        except OSError:
            print("Directorty cannot be opened in %s" % output_folder_path)

        impaired = '(IMPAIRED)';

        for f in listdir(input_folder_path):
            if impaired in f:
                # Parse hearing descriptions in subtitles

                input_subtitle = "%s/%s" % (input_folder_path, f)
                output_subtitle = "%s/%s" % (output_folder_path, f)

                with codecs.open(input_subtitle, 'r', encoding='utf-8', errors='ignore') as f:
                    # finds hearing descriptions
                    text = ' '.join(f.read().split('\n'))

            full_text.append(text)
        category_text = ' '.join(full_text)
        text_dict[category] = category_text

    return text_dict


category_dict = stem_subtitles(path.relpath("ProcessedSubtitles"), path.relpath("ProcessedSubtitles"))
cleaned_list = clean_stopword(category_dict['Adventure'])
#stemmed_data = stemming(cleaned_list)
plot_data(cleaned_list)


