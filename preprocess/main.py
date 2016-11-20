from os import path
from preprocess import preprocess_subtitles

def main():
    input_path = path.relpath("Subtitles")
    output_path = path.relpath("ProcessedSubtitles")
    preprocess_subtitles(input_path, output_path)

main()
