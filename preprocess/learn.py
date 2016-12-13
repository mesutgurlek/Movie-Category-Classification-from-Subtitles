from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from os import path
import numpy as np
from tokenization import *
import nltk

#  # Categorize words and plot them
category_dict = categorize_words(path.relpath("ProcessedNormalText"))
to_be_filtered = ['im', 'oh', 'dont', 'go', 'know', 'yeah', 'come', 'get', 'well']  # 'grunt', 'beep', 'grunts', ',', 'groan', 'speak', 'music']

# for i in categories:
#     for f in to_be_filtered:
#         category_dict[i] = category_dict[i].replace(f, '')
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


test_size = 200

# ProcessedNormalText has the whole data
# CategoryData has the hearing impared data
text, genre = tag_subtitles(path.relpath('ProcessedNormalText'))


for i in range(len(text)):
    for f in to_be_filtered:
        text[i] = text[i].replace(f, '')

# Initialize naive bayes object

acc_scores = []
alpha_values = np.arange(0.1, 2.0, 0.1)
alpha_values = [.1, .5, 1, 2, 3, 4, 5]
print(alpha_values)
for a in alpha_values:
    #clf = MultinomialNB(alpha=a) #naive bayes
    clf = LogisticRegression(C=a, max_iter= 1000)
    #clf = neighbors.KNeighborsClassifier( a * 10, 'distance') # knn
    #clf = svm.SVC() #support vector machine
    acc = 0
    print('.', a)
    sample_num = 10
    for i in range(sample_num):
        text, genre = randomize(text, genre)

        bow_tf = bag_of_words_and_tf(text)

        clf.fit(bow_tf[test_size:], genre[test_size:])

        test_data = bow_tf[:test_size]
        test_genre = genre[:test_size]

        predicted = clf.predict(test_data)
        acc += accuracy_score(test_genre, predicted)*100
    acc_scores.append(float(acc/sample_num))

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
