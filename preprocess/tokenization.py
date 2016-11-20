from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk
from os import path
from os import mkdir
from os import listdir
import codecs

text6 = "EmEditor is built to agilely handle files of any size. When you ask it to open a file over a certain size (you choose the size), EmEditor will automatically start using temporary disk space rather than clogging up your memory, unlike most other text editors, which try to keep the whole file in memory and ultimately fail. By default, EmEditor uses temporary files when it opens a file larger than 300 MB. You can check and edit this size in the Advanced tab of the Customize dialog box. If you open a file larger than this size, a few highlighting features are disabled, including multiple-line comments. Wrapping modes are also disabled for optimal speed. If you are opening a file larger than this size, make sure there is enough disk space in the temporary file folder. The default temporary folder is the system temporary folder, specified by the %TEMP% environment variable. You can override the temporary folder to any folder you would like, that has enough space available. EmEditor’s multithreaded design allows you to view documents during the opening of a large file. A status window appears during most time-consuming activities such as text editing, saving, searching, replacing, inserting and deleting, which allows you to monitor and cancel those activities at any time.me."


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


data = clean_stopword(text6)
data = stemming(data)
#print(data)

