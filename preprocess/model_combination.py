# coding=utf-8
# import knn_wpm_dpm
# import impaired_classification
import global_variables
from os import listdir
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from impaired_classification import ImpairedClassification


def main():
    # Hold f1 scores for each model
    f1_scores = []
    impaired_model = ImpairedClassification('TrainSubtitles', 'TestSubtitles')
    # impaired_model.tune_and_train()
    models = [impaired_model]

    f1 = impaired_model.get_f1_scores()
    f1_scores.append(f1)
    print(f1_scores)

    test_folder = 'CategoryDataTest'
    categories = global_variables.genres
    files = []
    y_true = []
    y_pred = []

    for category in categories:
        input_folder_path = "%s/%s" % (test_folder, category)
        print('.', )

        file_paths = []
        for f in listdir(input_folder_path):
            path = "%s/%s" % (input_folder_path, f)
            file_paths.append(path)

        for path in file_paths:
            y_true.append(category)
            files.append(path)

    shuffle(files, y_true)
    for filepath in files:
        predicted = impaired_model.predict(filepath)
        y_pred.append(predicted)
    print(classification_report(y_true, y_pred))

main()


