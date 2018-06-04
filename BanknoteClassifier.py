import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier

class BanknoteClassifier:

	def __init__(self, dataset):
		self.dataset_path = dataset
		df = pandas.read_csv(self.dataset_path, encoding="latin-1") # import the CSV dataset and save as a Pandas dataframe object
		print("The shape of the original dataset is {}".format(df.shape))

		targets = df["class"].values # save the targets (Y) as a NumPy array of integers
		print("The shape of y is {}".format(numpy.shape(targets)))

		features_df = df.loc[:, df.columns != "class"] # save everything else as the features (X)
		print("The shape of X is {}".format(numpy.shape(features_df)))

		features_names = features_df.columns.values # save the column names

		features_matrix = features_df.values # convert X from a dataframe to a matrix (for our machine learning models)

		scaled_features_matrix = StandardScaler().fit_transform(features_matrix) # scale the data so mean = 0, unit variance - important for many models

		[print("The mean is {} and standard deviation is {} for column {}.".format(round(numpy.mean(column),2), numpy.std(column), idx))
		 for idx, column in enumerate(scaled_features_matrix.T)] # check that the mean and std dev are 0 and 1

		self.training_features, self.test_features, self.training_class, self.test_class = train_test_split(scaled_features_matrix, targets, test_size=0.20, random_state=21)

		# check the shapes of the newly partitioned datasets
		print("The shape of the training features is {}".format(numpy.shape(self.training_features)))
		print("The shape of the test features is {}".format(numpy.shape(self.test_features)))
		print("The shape of the training targets is {}".format(numpy.shape(self.training_class)))
		print("The shape of the test targets is {}".format(numpy.shape(self.test_class)))

	def Classify(self, classifier):
		if classifier == 'linear':
			self.classifier = LinearSVC()
		elif classifier == 'random_forest':
			self.classifier = RandomForestClassifier(n_jobs = -1)
		elif classifier == 'neural_network':
			self.classifier = MLPClassifier(activation='logistic', solver='lbfgs', learning_rate='adaptive')
		elif classifier == 'gaussian':
			self.classifier = GaussianProcessClassifier(n_jobs = -1)
		elif classifier == 'knn':
			self.classifier = KNeighborsClassifier(n_jobs = -1)
		else:
			self.classifier = LogisticRegression()

		self.classifier.fit(self.training_features, self.training_class)
		training_predictions = self.classifier.predict(self.training_features)
		test_predictions = self.classifier.predict(self.test_features)

		data = {}

		data = {
			'training': {
				'total_count': len(self.training_class),
				'correct_prediction_count': numpy.sum(training_predictions == self.training_class),
				'accuracy': numpy.sum(training_predictions == self.training_class) / len(self.training_class)
			},
			'test': {
				'total_count': len(self.test_class),
				'correct_prediction_count': numpy.sum(test_predictions == self.test_class),
				'accuracy': numpy.sum(test_predictions == self.test_class) / len(self.test_class)
			}			
		}

		return data;