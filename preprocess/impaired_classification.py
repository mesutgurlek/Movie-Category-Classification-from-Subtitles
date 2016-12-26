from os import path
import codecs
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from sklearn.naive_bayes import MultinomialNB
from tokenization import tag_subtitles
from tokenization import randomize
from tokenization import bag_of_words_and_tf
from tokenization import clean_stopword
from tokenization import bag_of_words_and_tf
from preprocess import preprocess_subtitles
from tokenization import process_movie_subtitles
from preprocess import global_variables


class ImpairedClassification:
    """
        Impaired classification API
    """

    def __init__(self, train_path, test_path):
        self.clf = None
        self.vectorizor = None
        self.optimal_alpha = 0.01
        self.train_path = path.relpath(train_path)
        self.test_path = path.relpath(test_path)

    def tune_and_train(self):
        # Pre-process the training data
        train_output_path = path.relpath("ProcessedSubtitlesTrain")
        preprocess_subtitles(self.train_path, train_output_path)

        process_movie_subtitles(path.relpath("ProcessedSubtitlesTrain"), path.relpath("CategoryDataTrain"))

        train_text, train_genre = tag_subtitles(path.relpath('CategoryDataTrain'))

        to_be_filtered = ['grunt', 'beep', 'grunts', ',', 'groan', 'speak', 'music']
        for i in range(len(train_text)):
            for f in to_be_filtered:
                train_text[i] = train_text[i].replace(f, '')

        # Pre-process the test data
        test_output_path = path.relpath("ProcessedSubtitlesTest")
        preprocess_subtitles(self.test_path, test_output_path)

        process_movie_subtitles(path.relpath("ProcessedSubtitlesTest"), path.relpath("CategoryDataTest"))

        # Apply cross validation
        # alpha_values = np.arange(0.1, 2.0, 0.1)
        test_size = int((len(train_genre) * 20)/100)
        acc_scores = []
        alpha_values = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005]
        print(alpha_values)
        for a in alpha_values:
            clf_tmp = MultinomialNB(alpha=a)
            acc = 0
            print('.', a)
            for i in range(50):
                text, genre = randomize(train_text, train_genre)

                bow_tf, vectorizer = bag_of_words_and_tf(text)

                clf_tmp.fit(bow_tf[test_size:], genre[test_size:])

                test_data = bow_tf[:test_size]
                test_genre = genre[:test_size]

                predicted = clf_tmp.predict(test_data)
                acc += accuracy_score(test_genre, predicted)*100
            acc_scores.append(float(acc/50))
        self.optimal_alpha = alpha_values[acc_scores.index(max(acc_scores))]

    def get_f1_scores(self):
        train_text, train_genre = tag_subtitles(path.relpath('CategoryDataTrain'))
        test_size = int((len(train_genre) * 20) / 100)
        categories = global_variables.genres

        clf_tmp = MultinomialNB(alpha=self.optimal_alpha)
        text, genre = randomize(train_text, train_genre)

        bow_tf, vectorizor = bag_of_words_and_tf(text)

        clf_tmp.fit(bow_tf[test_size:], genre[test_size:])

        test_data = bow_tf[:test_size]
        test_genre = genre[:test_size]

        predicted = clf_tmp.predict(test_data)
        print('scikit learn score: ', accuracy_score(test_genre, predicted))
        print(classification_report(test_genre, predicted))
        p, r, f1, s = precision_recall_fscore_support(test_genre, predicted)
        f1_scores = [float("{0:.2f}".format(a)) for a in f1]
        f1_dict = {}

        # Initialize Model
        self.clf = MultinomialNB(alpha=self.optimal_alpha)
        bow_tf, self.vectorizor = bag_of_words_and_tf(text)
        self.clf.fit(bow_tf, genre)

        for idx, cat in enumerate(categories):
            f1_dict[cat] = f1_scores[idx]
        print(f1_dict)
        return f1_dict

    def predict(self, filepath):
        with codecs.open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # finds hearing descriptions
            text = ' '.join(f.read().split('\n'))
        bow_tf = self.vectorizor.transform([text])
        predicted = self.clf.predict(bow_tf)[0]
        return predicted

# def main():
#     input_path = path.relpath("TestSubtitles")
#     output_path = path.relpath("ProcessedSubtitlesTest")
#     preprocess_subtitles(input_path, output_path)
#
#     process_movie_subtitles(path.relpath("ProcessedSubtitlesTest"), path.relpath("CategoryDataTest"))
#
#     test_size = 300
#     test_text, test_genre = tag_subtitles(path.relpath('CategoryDataTest'))
#
#     to_be_filtered = ['grunt', 'beep', 'grunts', ',', 'groan', 'speak', 'music']
#     for i in range(len(test_text)):
#         for f in to_be_filtered:
#             test_text[i] = test_text[i].replace(f, '')
#
#     # Classification report for alpha: 0.01
#     clf = MultinomialNB(alpha=0.01)
#     text, genre = randomize(test_text, test_genre)
#
#     bow_tf = bag_of_words_and_tf(text)
#
#     clf.fit(bow_tf[test_size:], genre[test_size:])
#
#     test_data = bow_tf[:test_size]
#     test_genre = genre[:test_size]
#
#     predicted = clf.predict(test_data)
#     print('scikit learn score: ', accuracy_score(test_genre, predicted))
#     print(classification_report(test_genre, predicted))
#     p, r, f1, s = precision_recall_fscore_support(test_genre, predicted)
#     f1_scores = [float("{0:.2f}".format(a)) for a in f1]
#     print(f1_scores)
#
# main()
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

# process_movie_subtitles(path.relpath("ProcessedSubtitles"), path.relpath("CategoryData"))


# # Initialize naive bayes object
#
# acc_scores = []
# alpha_values = np.arange(0.1, 2.0, 0.1)
# alpha_values = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005]
# print(alpha_values)
# for a in alpha_values:
#     clf = MultinomialNB(alpha=a)
#     acc = 0
#     print('.', a)
#     for i in range(50):
#         text, genre = randomize(text, genre)
#
#         bow_tf = bag_of_words_and_tf(text)
#
#         clf.fit(bow_tf[test_size:], genre[test_size:])
#
#         test_data = bow_tf[:test_size]
#         test_genre = genre[:test_size]
#
#         predicted = clf.predict(test_data)
#         acc += accuracy_score(test_genre, predicted)*100
#     acc_scores.append(float(acc/50))
#
# print(acc_scores)
#
# plt.plot(alpha_values, acc_scores, 'o')
# plt.axis([0, 5, -1, 100])
#
# plt.xlabel('Alpha values')
# plt.ylabel('Accuracy')
# plt.legend(loc='upper right', numpoints=1)
# plt.title("Accuracies / Alpha values")
#
# #for k, accuracy in zip(k_values, accuracies):
# #    plt.text(k - 0.6, accuracy+1, str(k) + ", " + str(format(accuracy, '.1f')), fontsize=10)
#
# plt.show()

