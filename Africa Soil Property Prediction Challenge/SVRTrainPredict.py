import pickle
import pandas as pd
from sklearn.svm import SVR
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from cleanOrganise import cleanOrganise, targetSplit, createSubmission

## Reading in the training and testing data.
trainDF = pd.read_csv('training.csv')
testDF = pd.read_csv('sorted_test.csv')

## Splitting data.
X_Train, Y_Train = cleanOrganise(trainDF)
X_Test = cleanOrganise(testDF, test = True)

## Obtaining individual dataframes for the target variables as a list.
Y_TrainDict = targetSplit(Y_Train)

## Normalising the training and testing data. 
scaler = MinMaxScaler()

X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.fit_transform(X_Test)

## Running Support Vector Regression for each target variable.
yPred = {'PIDN' : testDF['PIDN'], 'Ca': None, 'P': None, 'pH': None, 'SOC': None, 'Sand' : None}
Classifiers = {'Ca': None, 'P': None, 'pH': None, 'SOC': None, 'Sand' : None}

for key, value in Y_Train.iteritems():
    
    print "Performing hyperparameter Tuning for : ", key
    
    ## Setting target variable.
    Y_Train = value
    
    ## Defining the hyperparameter space for searching the optimum set of hyperparameter values .
    parameters = { 
	               'kernel' : ['rbf','poly','sigmoid'],
	               'degree' : [3,4,5,6],
	               'gamma' : [0.01,0.02,0.03,0.10,0.20,0.30]
                  }

    ## Setting the classifier.
    Classifier = SVR(C = 10000.0)

    Clf = GridSearchCV(Classifier, parameters, n_jobs = 1, verbose = 5)

    ## Fitting the Model to the Training Data .
    Clf.fit(X_Train,Y_Train)
    
    # Choosing the best classifier parameters from the Grid Search results.
    Classifiers[key] = Clf.best_estimator_

## Performing the training with the best estimators obtained from the grid search.
for key, value in Classifiers.iteritems():

    print "Prediction for : ", key

    Classifier = Classifiers[key]

    Classifier.fit(X_Train, Y_TrainDict[key])
    
    ## Performing inference on the test data.
    yPred[key] = Classifier.predict(X_Test)

## Creating the CSV submission file.
submissionDataframe = createSubmission(yPred)
submissionDataframe.to_csv('SubmissionFiles/SVR.csv', index = False)

## Storing the hyper-parameter optimized classifiers.
with open('PreTrained/SVR.pickle', 'wb') as handle:
    pickle.dump(Classifiers, handle, protocol = pickle.HIGHEST_PROTOCOL)