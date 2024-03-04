import os
import json
import argparse
import matplotlib.pyplot as plt

def count_annotations(annotation_folder):
    annotation_counts = {}
    for root, _, files in os.walk(annotation_folder):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), 'r') as f:
                    annotations = f.readlines()
                    annotation_counts[file] = len(annotations)
    return annotation_counts

def generate_histogram(annotation_counts, subset, output_folder):
    histogram = {}
    for count in annotation_counts.values():
        if count not in histogram:
            histogram[count] = 1
        else:
            histogram[count] += 1

    plt.bar(histogram.keys(), histogram.values())
    plt.xlabel('Number of Annotations')
    plt.ylabel('Number of Images')
    plt.title(f'Annotation Distribution for {subset}')
    plt.savefig(os.path.join(output_folder, f'{subset}_histogram.png'))
    plt.show()

    return histogram

def accumulate_histogram(histograms, output_folder):
    accumulated_histogram = {}
    for histogram in histograms:
        for count, frequency in histogram.items():
            if count not in accumulated_histogram:
                accumulated_histogram[count] = frequency
            else:
                accumulated_histogram[count] += frequency

    plt.bar(accumulated_histogram.keys(), accumulated_histogram.values())
    plt.xlabel('Number of Annotations')
    plt.ylabel('Number of Images')
    plt.title('Accumulated Annotation Distribution')
    plt.savefig(os.path.join(output_folder, 'accumulated_histogram.png'))
    plt.show()

    return accumulated_histogram

def save_to_json(data, output_folder, subset):
    with open(os.path.join(output_folder, f'{subset}_annotation_counts.json'), 'w') as f:
        json.dump(data, f)

def main(input_folder, output_folder):
    subsets = ['train', 'test', 'val']
    histograms = []
    for subset in subsets:
        annotation_folder = os.path.join(input_folder, subset)
        annotation_counts = count_annotations(annotation_folder)
        histogram = generate_histogram(annotation_counts, subset, output_folder)
        histograms.append(histogram)
        save_to_json(histogram, output_folder, subset)

    accumulate_histogram(histograms, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process annotations and generate histograms.')
    parser.add_argument('--input_folder', type=str, help='Path to input folder containing train, test, and val subdirectories')
    parser.add_argument('--output_folder', type=str, help='Path to output folder to save JSON files and histograms')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)

