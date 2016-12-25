# coding=utf-8
from os import path
from os import mkdir
from os import listdir
import codecs
import global_variables
from random import shuffle
from shutil import copyfile


def prepare_dataset(input_folder, train_output_folder, test_output_folder):
    subtitles_path = path.relpath(input_folder)
    train_output_path = path.relpath(train_output_folder)
    test_output_path = path.relpath(test_output_folder)
    categories = global_variables.genres

    for category in categories:
        input_folder_path = "%s/%s" % (subtitles_path, category)
        train_output = "%s/%s" % (train_output_path, category)
        test_output = "%s/%s" % (test_output_path, category)

        print('.', )
        # Create folders for test and train outputs
        try:
            if not path.isdir(train_output):
                mkdir(train_output, 0o755)
        except OSError:
            print("Directory cannot be opened in %s" % train_output)

        try:
            if not path.isdir(test_output):
                mkdir(test_output, 0o755)
        except OSError:
            print("Directory cannot be opened in %s" % test_output)

        # Get IMPAIRED and srt files, store them in a list
        impaired = '(IMPAIRED)'
        files = []
        for f in listdir(input_folder_path):
            if impaired in f and f.endswith('.srt'):
                files.append(f)

        # Shuffle files and separate as %20 test and %80 train data
        print('******************* {} ***************'.format(category))
        shuffle(files)
        test_size = int((len(files) * 20) / 100)

        # Output the test dataset in TestSubtitles
        for f in files[:test_size]:
            input_subtitle = "%s/%s" % (input_folder_path, f)
            test_output_subtitle = "%s/%s" % (test_output, f)
            copyfile(input_subtitle, test_output_subtitle)

        # Output the train dataset in TrainSubtitles
        for f in files[test_size:]:
            input_subtitle = "%s/%s" % (input_folder_path, f)
            train_output_subtitle = "%s/%s" % (train_output, f)
            copyfile(input_subtitle, train_output_subtitle)


def main():
    input_path = 'Subtitles'
    train_output_path = 'TrainSubtitles'
    test_output_path = 'TestSubtitles'
    prepare_dataset(input_path, train_output_path, test_output_path)

main()
