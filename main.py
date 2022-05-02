import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn import svm

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Import datasets
englandDataFull = pd.read_csv('C:/Programming/cs350/data/englandDataFull.csv')

# Preparing dataset for classifier
englandData57 = englandDataFull.loc[englandDataFull['Year'].isin([2015, 2017])]
englandData9 = englandDataFull.loc[englandDataFull['Year'] == 2019]
# Train/Test Split
xTrain = englandData57.loc[:, [englandData57.columns[1]] + list(englandData57.columns[4:40])].values
yTrain = englandData57.iloc[:, 40].values
xTest = englandData9.loc[:, [englandData9.columns[1]] + list(englandData9.columns[4:40])].values
yTest = englandData9.iloc[:, 40].values

# Create random forest classifier
classifierRF = RandomForestClassifier(verbose=2)
classifierRF.fit(xTrain, yTrain)
# Tuning the random forest classifier
# Ranges for data
nEstimators = [int(x) for x in np.linspace(start=5, stop=2000, num=1)]
maxFeatures = ['auto', 'sqrt']
maxDepth = [int(x) for x in np.linspace(5, 100, num=1)]
maxDepth.append(None)
minSamplesSplit = [int(x) for x in np.linspace(start=2, stop=30, num=1)]
minSamplesLeaf = [int(x) for x in np.linspace(start=1, stop=30, num=1)]
bootstrap = [True, False]
randomGridRF = {'n_estimators': nEstimators,
               'max_features': maxFeatures,
               'max_depth': maxDepth,
               'min_samples_split': minSamplesSplit,
               'min_samples_leaf': minSamplesLeaf,
               'bootstrap': bootstrap}
# Fit the hyperparameters
tunedRF = RandomizedSearchCV(estimator=classifierRF, param_distributions=randomGridRF, n_iter=1000, cv=3, verbose=2, random_state=0, n_jobs=-1)
tunedRF.fit(xTrain, yTrain)

# Check accuracies
predictionsRF = classifierRF.predict(xTest)
baseRFAccuracy = accuracy_score(yTest, predictionsRF)
predictionsRFTuned = tunedRF.best_estimator_.predict(xTest)
tunedRFAccuracy = accuracy_score(yTest, predictionsRFTuned)

# Output into file
outF = open("params/paramsRF.txt", "a")
outF.write(str(tunedRF.best_params_))
print("\n")
outF.write("Tuned accuracy = " + str(tunedRFAccuracy))
print("\n")
outF.write("Base accuracy = " + str(baseRFAccuracy))
outF.close()

# Format results then output
fullRFTunedResults = englandData9
fullRFTunedResults["Prediction"] = predictionsRFTuned
fullRFBaseResults = englandData9
fullRFBaseResults["Prediction"] = predictionsRF
fullRFTunedResults.to_csv(r'C:/Programming/cs350/results/fullRFTunedResults.csv', index=False)
fullRFBaseResults.to_csv(r'C:/Programming/cs350/results/fullRFBaseResults.csv', index=False)

# Feature importances
importances = tunedRF.best_estimator_.feature_importances_
std = np.std([tree.feature_importances_ for tree in tunedRF.best_estimator_.estimators_], axis=0)
forestImportances = {"Feature": ([englandDataFull.columns[1]] + list(englandDataFull.columns[4:40])), "Importances": importances}
featuresRF = pd.DataFrame(data=forestImportances)
featuresRF.to_csv(r'C:/Programming/cs350/features/featuresRF.csv', index=False)

# # Permutation importances
# result = permutation_importance(tunedRF.best_estimator_, xTest, yTest, n_repeats=1, random_state=0, n_jobs=1)
# forestImportances = {"Feature": ([englandDataFull.columns[1]] + list(englandDataFull.columns[4:40])), "Importances": result.importances_mean}
# permutationRF = pd.DataFrame(data=forestImportances)
# # permutationRF.to_csv(r'C:/Programming/cs350/features/permutationRF.csv', index=False)

# Create SVM classifier
classifierSVM = svm.SVC(kernel='poly', verbose=2)
classifierSVM.fit(xTrain, yTrain)

# Tuning the svm classifier
# Ranges for data
kernels = ['sigmoid', 'rbf']
c =[0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
gamma = ['auto', 'scale']
paramGridSVM = {'kernel': kernels,
               'C': c,
               'gamma': gamma}
# Fit the hyperparameters
print("Creating tuned model")
tunedSVM = GridSearchCV(classifierSVM, paramGridSVM, cv=3, verbose=2, n_jobs=1)
print("Fitting tuned model")
tunedSVM.fit(xTrain, yTrain)
print("done tuning")

# Check accuracies
predictionsSVM = classifierSVM.predict(xTest)
baseSVMAccuracy = accuracy_score(yTest, predictionsSVM)
predictionsSVMTuned = tunedSVM.best_estimator_.predict(xTest)
tunedSVMAccuracy = accuracy_score(yTest, predictionsSVMTuned)

# Output into file
outF = open("params/paramsSVM.txt", "a")
outF.write(str(tunedSVM.best_params_))
print("\n")
outF.write("Tuned accuracy = " + str(tunedSVMAccuracy))
print("\n")
outF.write("Base accuracy = " + str(baseSVMAccuracy))
outF.close()

# Format results then output
fullSVMTunedResults = englandData9
fullSVMTunedResults["Prediction"] = predictionsSVMTuned
fullSVMBaseResults = englandData9
fullSVMBaseResults["Prediction"] = predictionsSVM
fullSVMTunedResults.to_csv(r'C:/Programming/cs350/results/fullSVMTunedResults.csv', index=False)
fullSVMBaseResults.to_csv(r'C:/Programming/cs350/results/fullSVMBaseResults.csv', index=False)

# # Permutation importances
# resultSVM = permutation_importance(tunedSVM.best_estimator_, xTest, yTest, n_repeats=1, random_state=0, n_jobs=-1)
# forestImportancesSVM = {"Feature": ([englandDataFull.columns[1]] + list(englandDataFull.columns[4:40])), "Importances": resultSVM.importances_mean}
# permutationSVM = pd.DataFrame(data=forestImportancesSVM)
# permutationSVM.to_csv(r'C:/Programming/cs350/features/permutationSVM.csv', index=False)


# # Create clusters and model on each
englandData57 = englandData57[englandData57.Year != 2017]
xTrain2017 = englandData57.loc[:, [englandData57.columns[1]] + list(englandData57.columns[4:40])].values
firstClassifierKM = KMeans(init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

visualizer = KElbowVisualizer(firstClassifierKM, k=(2, 30), timings=True)
visualizer.fit(xTrain2017)
visualizer.show()

classifierKM = KMeans(init='random', n_init=10, n_clusters=6, max_iter=300, tol=1e-04, random_state=0)
clusters = classifierKM.fit_predict(xTrain2017)
englandData57["Cluster"] = clusters

# RF with clusters
accuracies = 0
baseCRFResults = []
for i in range(6):
    # Get data
    partialData = englandData57.loc[englandData57['Cluster'] == i]
    xTrainCluster = partialData.loc[:, [partialData.columns[1]] + list(partialData.columns[4:40])].values
    yTrainCluster = partialData.iloc[:, 40].values
    partialTestData = englandData9.loc[englandData9['ONSConstID'].isin(partialData['ONSConstID'])]
    xTestCluster = partialTestData.loc[:, [partialTestData.columns[1]] + list(partialTestData.columns[4:40])].values
    yTestCluster = partialTestData.iloc[:, 40].values

    # Create classifier
    classifier = RandomForestClassifier(n_estimators=50, random_state=0)

    # Tuning each classifier
    # Ranges for data
    nEstimators = [int(x) for x in np.linspace(start=10, stop=2000, num=25)]
    maxFeatures = ['auto', 'sqrt']
    maxDepth = [int(x) for x in np.linspace(5, 100, num=5)]
    maxDepth.append(None)
    minSamplesSplit = [2, 5, 10]
    minSamplesLeaf = [1, 2, 4]
    bootstrap = [True, False]
    randomGridRF = {'n_estimators': nEstimators,
                    'max_features': maxFeatures,
                    'max_depth': maxDepth,
                    'min_samples_split': minSamplesSplit,
                    'min_samples_leaf': minSamplesLeaf,
                    'bootstrap': bootstrap}
    # Fit the hyperparameters
    tunedClusterRF = RandomizedSearchCV(estimator=classifier, param_distributions=randomGridRF, n_iter=100, cv=3, verbose=2,
                                 random_state=0, n_jobs=-1)
    tunedClusterRF.fit(xTrainCluster, yTrainCluster)
    predictions = tunedClusterRF.predict(xTestCluster)

    # Output predictions and accuracy
    partialTestData["Prediction"] = predictions
    results = partialTestData.values.tolist()
    baseCRFResults.extend(results)
    accuracies += (accuracy_score(yTestCluster, predictions))
# Send results to file
outF = open("params/paramsCRF.txt", "a")
outF.write(str(accuracies/6))
outF.close()
resultsCRF = pd.DataFrame(baseCRFResults, columns=partialTestData.columns)
resultsCRF.to_csv(r'C:/Programming/cs350/results/resultsCRF.csv', index=False)

# # SVM with clusters
# accuracies = 0
# baseCSVMResults = []
# for i in range(clustersAmount):
#     # Get data
#     partialData = englandData57.loc[englandData57['Cluster'] == i]
#     print(partialData.shape)
#     xTrainCluster = partialData.loc[:, [partialData.columns[1]] + list(partialData.columns[4:40])].values
#     yTrainCluster = partialData.iloc[:, 40].values
#     partialTestData = englandData9.loc[englandData9['ONSConstID'].isin(partialData['ONSConstID'])]
#     xTestCluster = partialTestData.loc[:, [partialTestData.columns[1]] + list(partialTestData.columns[4:40])].values
#     yTestCluster = partialTestData.iloc[:, 40].values
#
#     # Create classifier
#     classifier = svm.SVC(kernel='rbf')
#     classifier.fit(xTrainCluster, yTrainCluster)
#     predictions = classifier.predict(xTestCluster)
#
#     # Output predictions and accuracy
#     partialTestData["Prediction"] = predictions
#     results = partialTestData.values.tolist()
#     baseCSVMResults.extend(results)
#     accuracies += (accuracy_score(yTestCluster, predictions))
# #Send results to file
# print(accuracies/clustersAmount)
# resultsCSVM = pd.DataFrame(baseCSVMResults, columns=partialTestData.columns)
# resultsCSVM.to_csv(r'C:/Programming/cs350/results/resultsCSVM.csv', index=False)
# outF = open("params/paramsCSVM.txt", "w")
# outF.write(str(accuracies/clustersAmount))
# outF.close()


