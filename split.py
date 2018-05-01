import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # import a linear regression model
from sklearn.svm import LinearSVC # import a support vector machine with a linear kernel


df = pd.read_csv("data/dataset.csv", encoding="latin-1") # import the CSV dataset and save as a Pandas dataframe object
print("The shape of the original dataset is {}".format(df.shape))

targets = df["class"].values # save the targets (Y) as a NumPy array of integers
print("The shape of y is {}".format(np.shape(targets)))

features_df = df.loc[:, df.columns != "class"] # save everything else as the features (X)
print("The shape of X is {}".format(np.shape(features_df)))

features_names = features_df.columns.values # save the column names

features_matrix = features_df.values # convert X from a dataframe to a matrix (for our machine learning models)

scaled_features_matrix = StandardScaler().fit_transform(features_matrix) # scale the data so mean = 0, unit variance - important for many models

[print("The mean is {} and standard deviation is {} for column {}.".format(round(np.mean(column),2), np.std(column), idx))
 for idx, column in enumerate(scaled_features_matrix.T)] # check that the mean and std dev are 0 and 1

# test_size specifies % of samples to be allocated to test set
# random state is a seed to replicate results (use the same seed, we should all get the same results)
X_train, X_test, y_train, y_test = train_test_split(scaled_features_matrix, targets, test_size=0.20, random_state=21)

# check the shapes of the newly partitioned datasets
print("The shape of the training features is {}".format(np.shape(X_train)))
print("The shape of the test features is {}".format(np.shape(X_test)))
print("The shape of the training targets is {}".format(np.shape(y_train)))
print("The shape of the test targets is {}".format(np.shape(y_test)))

# fit the data to models
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
training_predictions = logistic_regression.predict(X_train)
test_predictions = logistic_regression.predict(X_test)

# check accuracy of Logistic Regression
number_correct = np.sum(training_predictions == y_train)
number_correct_test = np.sum(test_predictions == y_test)
print("\n\n {} out of {} correctly predicted on training set.".format(number_correct, len(y_train)))
print("That's an accuracy of {} for the Logistic Regression".format(number_correct / len(y_train)))
print("For the test set, we received an accuracy of {}".format(number_correct_test / len(y_test)))

linearSVM = LinearSVC()
linearSVM.fit(X_train, y_train)
training_predictions = linearSVM.predict(X_train)
test_predictions = linearSVM.predict(X_test)

# check accuracy of SVM
number_correct = np.sum(training_predictions == y_train)
number_correct_test = np.sum(test_predictions == y_test)
print("\n\n{} out of {} correctly predicted on training set.".format(number_correct, len(y_train)))
print("That's an accuracy of {} for the Linear SVM".format(number_correct / len(y_train)))
print("For the test set, we received an accuracy of {}".format(number_correct_test / len(y_test)))

# Random Forest

# Neural Network

# Gaussian Process Classifier

# KNN

# Other models?