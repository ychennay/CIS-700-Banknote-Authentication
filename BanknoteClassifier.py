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
        self.classifier = LogisticRegression()
        self.dataset_path = dataset
        self.test_size=0.20
        df = pandas.read_csv(self.dataset_path,
                             encoding="latin-1")  # import the CSV dataset and save as a Pandas dataframe object
        print("The shape of the original dataset is {}".format(df.shape))

        self.targets = df["class"].values  # save the targets (Y) as a NumPy array of integers
        print("The shape of y is {}".format(numpy.shape(self.targets)))

        features_df = df.loc[:, df.columns != "class"]  # save everything else as the features (X)
        print("The shape of X is {}".format(numpy.shape(features_df)))

        self.features_names = features_df.columns.values  # save the column names

        self.features_matrix = features_df.values  # convert X from a dataframe to a matrix (for our machine learning models)

        self.scaled_features_matrix = StandardScaler().fit_transform(
            self.features_matrix)  # scale the data so mean = 0, unit variance - important for many models

        [print("The mean is {} and standard deviation is {} for column {}.".format(round(numpy.mean(column), idx),
                                                                                   numpy.std(column), idx))
         for idx, column in enumerate(self.scaled_features_matrix.T)]  # check that the mean and std dev are 0 and 1

        self.training_features, self.test_features, self.training_class, self.test_class = train_test_split(
            self.scaled_features_matrix, self.targets, test_size=self.test_size, random_state=21)

        # check the shapes of the newly partitioned datasets
        print("The shape of the training features is {}".format(numpy.shape(self.training_features)))
        print("The shape of the test features is {}".format(numpy.shape(self.test_features)))
        print("The shape of the training targets is {}".format(numpy.shape(self.training_class)))
        print("The shape of the test targets is {}".format(numpy.shape(self.test_class)))

    def adjust_train_test_split(self, test_size, random_state=21):
        self.test_size=test_size
        self.training_features, self.test_features, self.training_class, self.test_class = train_test_split(
                self.scaled_features_matrix, self.targets, test_size=self.test_size, random_state=random_state)

    def Classify(self, classifier):
        if classifier == 'linear':
            self.classifier = LinearSVC()
        elif classifier == 'random_forest':
            self.classifier = RandomForestClassifier(n_jobs=-1)
        elif classifier == 'neural_network':
            self.classifier = MLPClassifier(activation='logistic', solver='lbfgs', learning_rate='adaptive')
        elif classifier == 'gaussian':
            self.classifier = GaussianProcessClassifier(n_jobs=-1)
        elif classifier == 'knn':
            self.classifier = KNeighborsClassifier(n_jobs=-1, weights='uniform', n_neighbors=12)

        self.classifier.fit(self.training_features, self.training_class)
        training_predictions = self.classifier.predict(self.training_features)
        test_predictions = self.classifier.predict(self.test_features)

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

        return data

    def CorruptData(self):

        negative_means = self.GetMean('training', 0)
        positive_means = self.GetMean('training', 1)

        training_df = pandas.DataFrame(self.training_features)
        training_positives_df = training_df[self.training_class == 1]

        current_distance = 9999
        closest_positive = None
        for i in range(training_positives_df.shape[0]):
            data_point = training_positives_df.values[i,:]
            distance = numpy.linalg.norm(data_point - negative_means)
            if distance < current_distance:
                closest_positive = data_point
                current_distance = distance

        print(f"Closest positive data point is {closest_positive}")

        distance = closest_positive - negative_means
        return closest_positive, distance, negative_means



    def GetMean(self, dataset, classification):

        """

        :param dataset: string option of either test or training
        :param classification: integer 0 or 1 to denote negative or positive
        :return:
        """
        mean = []
        for i in range(0, 3):
            mean.append(0)
        if dataset == 'test':
            class_source = self.test_class
            feature_source = self.test_features

        elif dataset == 'training':
            class_source = self.training_class
            feature_source = self.training_features
        targets = class_source == classification
        df = pandas.DataFrame(feature_source)
        filtered_features = df[targets].values
        means = numpy.mean(filtered_features, axis=0)
        print(f"The means for class {classification} are {means}")
        return means
