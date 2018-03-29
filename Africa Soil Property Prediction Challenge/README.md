# Africa Soil Property Prediction Challenge

Machine Learning solutions for the Kaggle competition **Africa Soil Property Prediction Challenge**. Two machine learning models were employed, namely, **Support Vector Regression** and **Random Forest Regression**.

The different files descriptions are as follows :

1. ``` training.csv ``` contains the training data.

2. ``` sorted_test.csv ``` contains the testing data.

3. ``` cleanOrganise.py ``` contains helper functions.

4. ``` RFRegressorTrainPredict.py ``` and ``` SVRTrainPredict.py ``` contain the code for performing hyper-parameter optimization for **Support Vector Regression** and **Random Forest Regression**, and perform prediction on the test set as well as store the optimized classifiers.

5. ``` RFRegressorPretrained.py ``` and ``` SVRPretrained.py ``` contain the code for loading the optimized classifiers for training and testing purposes.

## Usage

1. To perform the hyper-parameter optimization and store the resulting classifiers, run the following command.
(Caution : It might take considerable amount of time depending on the machine specifications)

``` python RFRegressorTrainPredict.py ``` or ``` python SVRTrainPredict.py ```

2. To load the hyper-parameter optimized model for performing predictions, run the following command.

``` python RFRegressorPretrained.py ``` and ``` python SVRPretrained.py ```

## Results

| Model                         | Private Score | Public Score   |
| ------------------------------|:-------------:| ---------------|
| Random Forest 				|  0.81704      | 0.81079		 |	    
| Support Vector Regression     |  0.80830      | 0.96002		 |
