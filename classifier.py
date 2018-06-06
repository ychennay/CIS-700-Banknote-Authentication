from BanknoteClassifier import *
import numpy as np

def print_data(type, data):
    print("{} set count: {}\nCorrectly predicted: {}\nAccuracy: {:.2f}%".format(type, data['total_count'],
                                                                                data['correct_prediction_count'],
                                                                                data['accuracy'] * 100))


if __name__ == "__main__":
    predictor = BanknoteClassifier(dataset='data/dataset.csv')
    # for each classifier of our choosing
    for cname in ['knn']:  # ['linear', 'random_forest', 'neural_network', 'gaussian', 'knn']:
        # interate 100 times, classifying, and corrupting the dataset using previous classification results

        result = predictor.Classify(classifier=cname)
        closest_positive, distance, negative_mean = predictor.CorruptData()
        STEPS = 50
        for idx in range(0, STEPS):

            step = -distance / STEPS
            print("before closest positive point:", closest_positive)
            closest_positive += step
            print("after, closest positive point:", closest_positive)
            print("negative mean:", negative_mean)
            print("distance from negative centroid: ", numpy.linalg.norm(closest_positive - negative_mean))

            preds = predictor.classifier.predict(closest_positive.reshape(1,-1))
            print(f"Prediction", preds)
            if preds[0] == 1: # only if we receive a positive classification
                print(predictor.training_features.shape)
                new_feature_space = np.vstack((predictor.training_features, closest_positive))
                new_class = np.append(predictor.training_class, 1)
                print("New class shape:", new_class.shape)
                print("New feature space", new_feature_space.shape)
                predictor.training_features = new_feature_space
                predictor.training_class = new_class
                predictor.classifier.fit(new_feature_space, new_class)
                input()
            else: # attacker is caught!
                print("Classified as negative :(")
                break

        print("\n\nClassifier: {}\n------------------------------".format(cname))
        print_data('Training', result['training'])
        print_data('Test', result['test'])
