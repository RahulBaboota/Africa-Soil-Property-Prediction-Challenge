import pickle
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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
    
    print "Performing hyperparameter Tuning for  : ", key
    
    ## Setting target variable.
    Y_Train = value
    
    ## Defining the hyperparameter space for searching the optimum set of hyperparameter values .
    parameters = { 'n_estimators' : [150,200],
               'max_features' : ['sqrt','log2'],
               'max_depth' : [5,8,10],
               'min_samples_split' : [2,5,10,15,100]
              }

    ## Setting the classifier.
    Classifier = RandomForestRegressor()

    Clf = GridSearchCV(Classifier, parameters, n_jobs = 5, verbose = 5)

    ## Fitting the Model to the Training Data .
    Clf.fit(X_Train,Y_Train)
    
    # Choosing the best classifier parameters from the Grid Search results.
    Classifiers[key] = Clf.best_estimator_

## Performing the training with the best estimators obtained from the grid search.
for key, value in Classifiers.iteritems():

    print "Training and Prediction for : ", key

    Classifier = Classifiers[key]

    Classifier.fit(X_Train, Y_TrainDict[key])
    
    ## Performing inference on the test data.
    yPred[key] = Classifier.predict(X_Test)

## Creating the CSV submission file.
submissionDataframe = createSubmission(yPred)
submissionDataframe.to_csv('SubmissionFiles/RFRegressor.csv', index = False)

## Storing the hyper-parameter optimized classifiers.
with open('PreTrained/RFRegressor.pickle', 'wb') as handle:
    pickle.dump(Classifiers, handle, protocol = pickle.HIGHEST_PROTOCOL)