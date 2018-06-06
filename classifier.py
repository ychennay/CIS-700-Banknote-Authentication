from BanknoteClassifier import *


def print_data(type, data):
    print("{} set count: {}\nCorrectly predicted: {}\nAccuracy: {:.2f}%".format(type, data['total_count'],
                                                                                data['correct_prediction_count'],
                                                                                data['accuracy'] * 100))


if __name__ == "__main__":
    predictor = BanknoteClassifier(dataset='data/dataset.csv')
    # for each classifier of our choosing
    for cname in ['knn']:  # ['linear', 'random_forest', 'neural_network', 'gaussian', 'knn']:
        # interate 100 times, classifying, and corrupting the dataset using previous classification results
        for idx in range(0, 99):
            result = predictor.Classify(classifier=cname)
            predictor.CorruptData()
        print("\n\nClassifier: {}\n------------------------------".format(cname))
        print_data('Training', result['training'])
        print_data('Test', result['test'])
