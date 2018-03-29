import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from cleanOrganise import cleanOrganise, targetSplit, createSubmission

# Reading in the training and testing data.
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

## Loading the hyper-parameter optimized classifiers.
with open('PreTrained/SVR.pickle', 'rb') as handle:
    Classifiers = pickle.load(handle)

yPred = {'PIDN' : testDF['PIDN'], 'Ca': None, 'P': None, 'pH': None, 'SOC': None, 'Sand' : None}

# Performing the training with the best estimators obtained from the grid search.
for key, value in Classifiers.iteritems():

    print "Prediction for : ", key
   
    Classifier = Classifiers[key]

    Classifier.fit(X_Train, Y_TrainDict[key])
    
    ## Performing inference on the test data.
    yPred[key] = Classifier.predict(X_Test)

## Creating the CSV submission file.
submissionDataframe = createSubmission(yPred)
submissionDataframe.to_csv('SubmissionFiles/SVR.csv', index = False)