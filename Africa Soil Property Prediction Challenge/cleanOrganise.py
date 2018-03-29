import pandas as pd

## Defining a function to clean and organise the data.
def cleanOrganise(Dataframe, test = False):
    
    ## One-Hot Encoding the Depth fields.
    Dataframe = Dataframe.replace(['Topsoil', 'Subsoil'], [0,1])
    
    ## Drop the C02 bands.
    toDrop = Dataframe.loc[:,'m2379.76':'m2352.76'].columns
    Dataframe.drop(labels = toDrop, axis = 1, inplace = True)

    ## Drop the PIDN column.
    Dataframe.drop(labels = 'PIDN', axis = 1, inplace = True)
      
    if (test == True):
        
        return Dataframe
    
    else:
        
        ## Obtaining the target variables.
        target = Dataframe.loc[:, 'Ca':'Sand']

        ## Dropping target variables.
        Dataframe.drop(labels = ['Ca', 'P', 'pH', 'SOC', 'Sand'], axis = 1, inplace = True)
    
        # return infrared, spatial, target
        return Dataframe, target

## Defining a function to return the individual target variables.
def targetSplit(Y):

    ## Creating a dictionary to hold the indiviudal target variables.
    yDict = {'Ca' : Y['Ca'],
             'P' : Y['P'],
             'pH' : Y['pH'],
             'SOC' : Y['SOC'],
             'Sand' : Y['Sand']}

    return yDict

## Defining a function to create the submission file.
def createSubmission(yPred):

    dataFrame = pd.DataFrame(yPred)

    return dataFrame[['PIDN', 'Ca', 'P', 'pH', 'SOC', 'Sand']]