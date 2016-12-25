from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem.porter import *

from os import path
from os import mkdir
from os import listdir
import codecs
import global_variables


def clean_stopword(text):
    stop = set(stopwords.words('english'))
    normalized = [i for i in text.lower().split() if i not in stop]
    return normalized


def stemming(text_array):
    stemmer = PorterStemmer()
    result = [stemmer.stem(text) for text in text_array]
    return result


def categorize_words(input_folder):
    subtitles_path = path.relpath(input_folder)

    text_dict = {}
    for category in global_variables.genres:

        text = ""
        full_text = []
        input_folder_path = "%s/%s" % (subtitles_path, category)\

        impaired = '(IMPAIRED)'
        for f in listdir(input_folder_path):
            if impaired in f:
                # Parse hearing descriptions in subtitles

                input_subtitle = "%s/%s" % (input_folder_path, f)

                with codecs.open(input_subtitle, 'r', encoding='utf-8', errors='ignore') as f:
                    # finds hearing descriptions
                    text = ' '.join(f.read().split('\n'))

            full_text.append(text)
        category_text = ' '.join(full_text)
        text_dict[category] = category_text

    return text_dict


def process_movie_subtitles(input_folder, output_folder):
    subtitles_path = path.relpath(input_folder)
    output_path = path.relpath(output_folder)

    for category in global_variables.genres:

        input_folder_path = "%s/%s" % (subtitles_path, category)
        output_folder_path = "%s/%s" % (output_path, category)
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

                with codecs.open(input_subtitle, 'r', encoding='utf-8', errors='ignore') as f:
                    # finds hearing descriptions
                    text = ' '.join(f.read().split('\n'))

                #Cleans stopwords and calls stemming
                processed_word_list = stemming(clean_stopword(text))
                new_text = '\n'.join(processed_word_list)

                with codecs.open(output_subtitle, 'w', encoding='utf-8', errors='ignore') as f:
                    # finds hearing descriptions
                    f.write(new_text)


def tag_subtitles(input_folder):
    # Returns a 2d list which data[i][0] gives tag, data[i][1] gives text of i'th movie
    subtitles_path = path.relpath(input_folder)
    genres = []
    texts = []

    # get lower bound and put equal amount of categories in the train set.
    lowest = 100000
    for category in global_variables.genres:
        input_folder_path = "%s/%s" % (subtitles_path, category)
        if len(listdir(input_folder_path)) < lowest:
            lowest = len(listdir(input_folder_path))

    print(lowest * len(global_variables.genres))

    for category in global_variables.genres:
        input_folder_path = "%s/%s" % (subtitles_path, category)
        impaired = '(IMPAIRED)'

        count = 0
        for f in listdir(input_folder_path):
            if impaired in f:
                if count > lowest:
                    break
                count += 1

                # Parse hearing descriptions in subtitles
                input_subtitle = "%s/%s" % (input_folder_path, f)
                with codecs.open(input_subtitle, 'r', encoding='utf-8', errors='ignore') as f:
                    # finds hearing descriptions
                    text = ' '.join(f.read().split('\n'))
                genres.append(category)
                texts.append(text)
    return texts, genres


def bag_of_words_and_tf(data):
    tf_bow_vectorizor = TfidfVectorizer()
    train_tf = tf_bow_vectorizor.fit_transform(data)

    return train_tf, tf_bow_vectorizor


def randomize(text, genre):
    return shuffle(text, genre)


def filter_words(text):
    to_be_filtered = ["grunt", "beep", "grunts", ",", "groan", "speak", "music"]
    for _i, movie in enumerate(text):
        for f in to_be_filtered:
            text[_i].lower().replace(f, '')
    return text



