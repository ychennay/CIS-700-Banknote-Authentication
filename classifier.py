from BanknoteClassifier import *

def print_data(type, data):
	print("{} set count: {}\nCorrectly predicted: {}\nAccuracy: {:.2f}%".format(type, data['total_count'], data['correct_prediction_count'], data['accuracy'] * 100))

if __name__ == "__main__":
	predictor = BanknoteClassifier(dataset= 'data/dataset.csv')
	for cname in ['linear', 'random_forest', 'neural_network', 'gaussian', 'knn']:
		result = predictor.Classify(classifier=cname)
		print("\n\nClassifier: {}\n------------------------------".format(cname))
		print_data('Training', result['training'])
		print_data('Test', result['test'])