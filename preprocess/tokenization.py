from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.metrics import accuracy_score
from os import path
from os import mkdir
from os import listdir
import codecs
import matplotlib.pyplot as plt
import numpy as np

categories = ['Action', 'Adventure', 'Comedy', 'Horror', 'Romance', 'War']


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
    for category in categories:

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

    for category in categories:

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
    for category in categories:
        input_folder_path = "%s/%s" % (subtitles_path, category)
        impaired = '(IMPAIRED)'
        for f in listdir(input_folder_path):
            if impaired in f:
                # Parse hearing descriptions in subtitles
                input_subtitle = "%s/%s" % (input_folder_path, f)
                with codecs.open(input_subtitle, 'r', encoding='utf-8', errors='ignore') as f:
                    # finds hearing descriptions
                    text = ' '.join(f.read().split('\n'))
                genres.append(category)
                texts.append(text)
    return texts, genres


def bag_of_words_and_tf(data):
    # Vectorize
    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(data)

    #Tf transform
    tf_transformer = TfidfTransformer(use_idf=False).fit(train_counts)
    train_tf = tf_transformer.transform(train_counts)

    return train_tf


def randomize(text, genre):
    return shuffle(text, genre)


def filter_words(text):
    to_be_filtered = ["grunt", "beep", "grunts", ",", "groan", "speak", "music"]
    for _i, movie in enumerate(text):
        for f in to_be_filtered:
            text[_i].lower().replace(f, '')
    return text

#  # Categorize words and plot them
# category_dict = categorize_words(path.relpath("CategoryData"))
# to_be_filtered = ['grunt', 'beep', 'grunts', ',', 'groan', 'speak', 'music']
#
# # for i in categories:
# #     for f in to_be_filtered:
# #         category_dict[i] = category_dict[i].replace(f, '')
#
# for c in categories:
#     cleaned_list = clean_stopword(category_dict[c])
#     stemmed_data = stemming(cleaned_list)
#
#     fd = nltk.FreqDist(stemmed_data)
#     print('Category: ', c)
#     print(fd.most_common(12))
#     fd.plot(12, cumulative=False)

# process_movie_subtitles(path.relpath("ProcessedSubtitles"), path.relpath("CategoryData"))w


test_size = 150

text, genre = tag_subtitles(path.relpath('CategoryData'))

to_be_filtered = ['grunt', 'beep', 'grunts', ',', 'groan', 'speak', 'music']
for i in range(len(text)):
    for f in to_be_filtered:
        text[i] = text[i].replace(f, '')

# Initialize naive bayes object

acc_scores = []
alpha_values = np.arange(0.1, 2.0, 0.1)
alpha_values = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005]
print(alpha_values)
for a in alpha_values:
    clf = MultinomialNB(alpha=a)
    acc = 0
    print('.', a)
    for i in range(50):
        text, genre = randomize(text, genre)

        bow_tf = bag_of_words_and_tf(text)

        clf.fit(bow_tf[test_size:], genre[test_size:])

        test_data = bow_tf[:test_size]
        test_genre = genre[:test_size]

        predicted = clf.predict(test_data)
        acc += accuracy_score(test_genre, predicted)*100
    acc_scores.append(float(acc/50))

print(acc_scores)

plt.plot(alpha_values, acc_scores, 'o')
plt.axis([0, 5, -1, 100])

plt.xlabel('Alpha values')
plt.ylabel('Accuracy')
plt.legend(loc='upper right', numpoints=1)
plt.title("Accuracies / Alpha values")

#for k, accuracy in zip(k_values, accuracies):
#    plt.text(k - 0.6, accuracy+1, str(k) + ", " + str(format(accuracy, '.1f')), fontsize=10)

plt.show()
# true_data = 0
# for i in range(test_size):
#     if predicted[i] == test_genre[i]:
#         true_data += 1
#     print('Predicted: {}, Original: {}'.format(predicted[i], test_genre[i]))
# print("overall accuracy: ", float(true_data/test_size))

#print('scikit learn score: ', accuracy_score(test_genre, predicted))

