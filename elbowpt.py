from BanknoteClassifier import *
import json
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mutual_info_score, recall_score


def print_data(type, data):
    print("{} set count: {}\nCorrectly predicted: {}\nAccuracy: {:.2f}%".format(type, data['total_count'],
                                                                                data['correct_prediction_count'],
                                                                                data['accuracy'] * 100))


if __name__ == "__main__":
    cl = BanknoteClassifier(dataset='data/dataset.csv')
    plotthis = []
    # for each classifier of our choosing
    for cname in ['knn', 'linear', 'random_forest', 'neural_network', 'gaussian']:
        # interate 100 times, classifying, and corrupting the dataset using previous classification results
        split = []
        train_accuracy = []
        test_accuracy = []
        print("Classifier: {}".format(cname.upper()))
        i=0
        for trainset in range(20, 100, 3):
            i += 1
            cl.adjust_train_test_split(trainset*0.01)
            result = cl.Classify(classifier=cname)
            # ---- iterative debugging ----
            # print("--- Iteration {} ---".format(i))
            # print_data('Training', result['training'])
            # print_data('Test', result['test'])
            # ---- json printout ----
            run = {
                'classifier': cname,
                'train/test split': '{}%'.format(trainset),
                'train set': result['training']['total_count'],
                'train accuracy' : result['training']['accuracy']*100,
                'test set' : result['test']['total_count'],
                'test accuracy' : result['test']['accuracy']*100
            }

            # ---- X/Y plotting ----
            split.append(trainset)
            train_accuracy.append(run['train accuracy'])
            test_accuracy.append(run['test accuracy'])
            print(json.dumps(run, indent=3))

        plot.plot(split, train_accuracy, 'ro', split, test_accuracy, 'g^')
        plot.xlabel('Split ratio')
        plot.ylabel('Accuracy')
        plot.show()
        input()
