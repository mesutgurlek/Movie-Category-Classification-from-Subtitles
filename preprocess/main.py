from os import path
from preprocess import preprocess_subtitles
from tokenization import process_movie_subtitles
from preprocessNormalText import preprocess_normal_text


def main():
    input_path = path.relpath("Subtitles")
    output_path = path.relpath("ProcessedSubtitles")
    preprocess_subtitles(input_path, output_path)

    process_movie_subtitles(path.relpath("ProcessedSubtitles"), path.relpath("CategoryData"))

    # in_path = path.relpath("Subtitles")
    # out_path = path.relpath("ProcessedNormalText")
    #
    # preprocess_normal_text(in_path, out_path)


main()
