from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk
from os import path
from os import mkdir
from os import listdir
import codecs
categories = ['Action', 'Adventure', 'Comedy', 'Horror', 'Romance', 'War']


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

    return shuffle(text, genre, random_state=100)

# # Categorize words and plot them
# category_dict = categorize_words(path.relpath("ProcessedSubtitles"))
# for c in categories:
#     cleaned_list = clean_stopword(category_dict[c])
#     stemmed_data = stemming(cleaned_list)
#     plot_data(stemmed_data)


#process_movie_subtitles(path.relpath("ProcessedSubtitles"), path.relpath("CategoryData"))
test_size = 200

text, genre = tag_subtitles(path.relpath('CategoryData'))
text, genre = randomize(text,genre)

bow_tf = bag_of_words_and_tf(text)
clf = MultinomialNB().fit(bow_tf[test_size:], genre[test_size:])

total_adv = 0
for i in genre[test_size:]:
    if i == 'Adventure':
        total_adv +=1

print( "total adv", total_adv)
print( "total movies", len(genre[test_size:]))



test_data = bow_tf[:test_size]
test_genre = genre[:test_size]
predicted = clf.predict(test_data)

print(predicted)

true_data = 0
for i in range(test_size):
    if predicted[i] == test_genre[i]:
        true_data += 1
    print('Predicted: {}, Original: {}'.format(predicted[i], test_genre[i]))

print("overall accuracy: ", float(true_data/test_size))




