#!/usr/bin/env python
# coding: utf-8


import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn import metrics
import pickle
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class Classifier():
    def __init__(self, recordings_folder, model_file_path="model", num_models=1):
        self.recordings_folder = recordings_folder
        self.model_file_path = recordings_folder + "\\" + model_file_path
        self.models = [None] * num_models
        self.num_models = num_models
    
    def read_train_test_data(self, X_file, y_file):  # call this function after running MI4
        X = scipy.io.loadmat(os.path.join(self.recordings_folder, X_file))[X_file]
        y = scipy.io.loadmat(os.path.join(self.recordings_folder, y_file))[y_file].T
        data = np.concatenate((X, y), axis=1)

        np.random.shuffle(data)

        data_one = data[np.where(data[:, -1] == 1)]
        data_two = data[np.where(data[:, -1] == 2)]
        data_three = data[np.where(data[:, -1] == 3)]

        num_of_trials = min(data_one.shape[0], data_two.shape[0], data_three.shape[0])

        data_one = data_one[:num_of_trials]
        data_two = data_two[:num_of_trials]
        data_three = data_three[:num_of_trials]

        n_train = int(np.floor(len(data_one) * 0.75))

        training_one, test_one = data_one[:n_train, :], data_one[n_train:, :]
        training_two, test_two = data_two[:n_train, :], data_two[n_train:, :]
        training_three, test_three = data_three[:n_train, :], data_three[n_train:, :]

        training = np.concatenate((training_one, training_two, training_three), axis=0)
        test = np.concatenate((test_one, test_two, test_three), axis=0)
        np.random.shuffle(training)
        np.random.shuffle(test)

        x_train = training[:, :-1]
        y_train = training[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]
        return x_train, y_train, x_test, y_test  # returned as numpy arrays

    def read_dataset(self, featuresFileName, labelsFileName):   # call this function after running MI4
        X = scipy.io.loadmat(os.path.join(self.recordings_folder, featuresFileName))[featuresFileName]
        y = scipy.io.loadmat(os.path.join(self.recordings_folder, labelsFileName))[labelsFileName]
        return X, y  # returned as numpy arrays

    def save_model(self, model):
        try:
            with open(self.model_file_path, 'wb') as f:
                pickle.dump(model, f, protocol=4)
                print("Saved model to file named {}".format(self.model_file_path))
        except BaseException as e:
            print("Exception while trying to save model to file{}: \n Exception:{}".format(self.model_file_path, e))

    def save_multiple_models(self, models):
        for i, model in enumerate(models):
            try:
                model_path = self.model_file_path + str(i)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f, protocol=4)
                    print("Saved model to file named {}".format(model_path))
            except BaseException as e:
                print("Exception while trying to save model to file {}: \n Exception:{}".format(model_path, e))


    def load_model(self):
        for i in range(self.num_models):
            try:
                if self.num_models==1:
                    path =self.model_file_path
                else:
                    path = self.model_file_path + str(i)
                with open(path, 'rb') as f:
                    self.models[i] = pickle.load(f)
            except BaseException as e:
                print("Exception while trying to read model from file {}: \n Exception:{}".format(self.model_file_path, e))

    def load_multiple_models(self, num_models=2):
        self.models = []
        for i in range(num_models):
            try:
                with open(self.model_file_path + str(i), 'rb') as f:
                    self.models.append(pickle.load(f))
            except BaseException as e:
                print("Exception while trying to read model from file {}: \n Exception:{}".format(self.model_file_path, e))

    def load_data(self, features_file_name, labels_file_name):
        features = scipy.io.loadmat(os.path.join(self.recordings_folder, features_file_name))[features_file_name]
        labels = scipy.io.loadmat(os.path.join(self.recordings_folder, labels_file_name))[labels_file_name].squeeze()
        return features, labels  # returned as numpy arrays

    def train_multiclass_model(self, X_train, y_train):
        clf = RandomForestClassifier.fit(X_train, y_train)
        return clf

    def predict_class(self, datapoint, pred_details_flag=False):
        """
        Get prediction's value for a single datapoint (trial's features)
        """
        if self.models is None and self.num_models==1:
            self.load_model()
        if self.num_models == 2:
            clf_INI, clf_LR = self.models
            preds_INI = clf_INI.predict(datapoint)
            preds_LR  = clf_LR.predict(datapoint)
            preds = np.zeros(preds_INI.shape)
            preds[preds_INI == 1] = 1
            preds[preds_INI == 4] = preds_LR[preds_INI == 4]
            if pred_details_flag:
                return preds, preds_INI, preds_LR
            else:
                return preds
        else:
            preds = self.models[0].predict(datapoint)
        return preds

def double_binary_fit(classifiers, X_train, y_train):
    clf_INI, clf_LR = classifiers
    y_train_INI = y_train.copy()
    y_train_INI[np.logical_or(y_train_INI == 2, y_train_INI == 3)] = 4
    clf_INI.fit(X_train, y_train_INI)
    train_LR_filter = y_train != 1
    y_train_LR, X_train_LR = y_train[train_LR_filter], X_train[train_LR_filter]
    clf_LR.fit(X_train_LR, y_train_LR)
    return clf_INI, clf_LR


def plot_confusion_matrix(labels, predictions, classe_names, title):  
    cf_matrix = confusion_matrix(labels, predictions)
    cf_matrix = cf_matrix / np.sum(cf_matrix)
    plt.figure()
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title(title);
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(classes_names)
    ax.yaxis.set_ticklabels(classes_names)


def predict(classifier, datapoints):
    classifier.load_model()
    datapoints = np.array(datapoints)
    
    if datapoints.ndim == 1:
        datapoints = datapoints.reshape(1, -1)
        
    prediction = classifier.predict_class(datapoints)
    return prediction


if __name__ == '__main__':

    # parameters:
    #   action: train/predict/committee_predict/test_performance
    #   featuresVariable: name of data file with features (e.g., AllDataInFeatures / AllDataTopFeatures)
    #   num_models: 1 - multiclass classifier / 2 - double binary classifier
    #               Required if action is train/predict/committee_predict
    #   recfolder: folder with training data / saved trained model. 
    #              Required if action is train/predict/test_performance.
    #   recfolderlist: list of folders with training data.
    #                  Required if action is committee_predict
    #   datapoints: datapoints (in features) to predict. 
    #               Required if action is predict/committee_predict
    #   show_CM: 0 - plot confusion matrix / 1 - don't plot confusion matrix
    #            Required if action is test_performance

    
    if action != "test_performance":
        num_models = int(num_models)
    if action in ["train", "predict"]:
        classifier = Classifier(recordings_folder=recfolder, num_models=num_models)
    if action == "train":
        # train model on data 

        X_file = featuresVariable
        y_file = r'AllDataLabels'   # name of data file with labels

        X_train, y_train = classifier.load_data(X_file, y_file)

        if num_models == 1:
            clf = RandomForestClassifier(n_estimators=50)
            clf.fit(X_train, np.ravel(y_train.T))
            classifier.save_model(clf)
            result = "Success"
        elif num_models == 2:
            classifiers = (RandomForestClassifier(n_estimators=50), RidgeClassifier())
            classifiers = double_binary_fit(classifiers, X_train, y_train)
            classifier.save_multiple_models(classifiers)
            result = "Success"
        else:
             print(f"Unsupported number of models: {num_models}\nSupported num_models are 1 (multiclass) or 2 (double binary)")


    elif action == "predict":
        # predict using saved trained model

        prediction = predict(classifier, datapoints)


    elif action == "committee_predict":
        # train models on each recording set and predict according to majority vote
        
        recfolderlist = recfolderlist.split(";")
        nm = len(recfolderlist)
        if np.array(datapoints).ndim == 1:
            nd = np.array(datapoints)[np.newaxis,:].shape[0]
        else:
            nd = np.array(datapoints).shape[0]
        predictionslist = np.zeros([nm,nd], dtype=int)
        prediction = np.zeros(nd)
        for i, recfolder in enumerate(recfolderlist):
            classifier = Classifier(recordings_folder=recfolder, num_models=num_models)
            predictionslist[i,:] = predict(classifier, datapoints)
        for i in range(nd):
            all_votes = np.bincount(predictionslist[:,i])
            maximum = max(all_votes)
            winners = (all_votes == maximum)
            if sum(winners) == 1:
                prediction[i] = all_votes.argmax()
            elif all(winners[1:]):
                # all classes are tied - choose randomly
                prediction[i] = np.random.choice(predictionslist[:,i])
            elif winners[1]:
                # idle was chosen along with left/right - choose left/right 
                # (since right/left was also chosen and they are similar)
                prediction[i] = winners[2:].argmax()
            else:
                # left and right are tied - choose left 
                # (since the models are less likely to predict left)
                prediction[i] = 2

    elif action == "test_performance":
        # test performance of classifier by splitting the data into train and test num_reps times
        
        X_file = r'AllDataInFeatures'
        y_file = r'AllDataLabels'
        num_reps = 100
        n_estimators = 50
        accurecies = [[[] for i in range(num_reps)] for j in range(2)]
        classifier = Classifier(recordings_folder=recfolder)
        labels_all = []
        predictions_MC_all = []
        predictions_DB_all = []
        predictions_INI_all = []
        predictions_LR_all = []
        for i in range(num_reps):
            print(f"iter = {i}")
            X_train, y_train, X_test, y_test = classifier.read_train_test_data(X_file, y_file)
            # select features
            selector = SelectKBest(score_func=f_classif, k=10)
            fit = selector.fit(X_train, y_train)
            X_train = fit.transform(X_train)
            X_test = fit.transform(X_test)
            labels_all = [*labels_all, *y_test]
            for num_models in range(1, 3):
                classifier = Classifier(recordings_folder=recfolder, num_models=num_models)
                if num_models == 1:
                    classifier0 = RandomForestClassifier(n_estimators=n_estimators)
                    classifier.models[0] = classifier0.fit(X_train, y_train)
                    y_test_predicted = classifier.predict_class(X_test)
                else:
                    classifiers = [RandomForestClassifier(n_estimators=n_estimators), RidgeClassifier()]
                    classifier.models = double_binary_fit(classifiers, X_train, y_train)
                    y_test_predicted, preds_INI, preds_LR = classifier.predict_class(X_test, pred_details_flag = True)
                if num_models == 1:
                    predictions_MC_all = [*predictions_MC_all, *y_test_predicted]
                else:
                    predictions_DB_all = [*predictions_DB_all, *y_test_predicted]
                    predictions_INI_all = [*predictions_INI_all, *preds_INI]
                    predictions_LR_all = [*predictions_LR_all, preds_LR]
                accurecies[num_models-1][i].append(metrics.accuracy_score(y_test.T, y_test_predicted))
        print(f'accurecies = {accurecies}')
        means = np.array(accurecies).mean(axis=1)
        if show_CM:
            # plot confusion matrix
            classes_names = ['Idle', 'Left', 'Right']
            title = 'Multiclass'
            plot_confusion_matrix(labels_all, predictions_MC_all, classes_names, title)
            title = 'Double binary'
            plot_confusion_matrix(labels_all, predictions_DB_all, classes_names, title)
            # Plot each binary 
            classes_names = ['Idle', 'Not Idle']
            title = 'Binary'
            labels_all_INI = np.array(labels_all.copy())
            labels_all_INI[np.logical_or(labels_all_INI == 2, labels_all_INI == 3)] = 4
            plot_confusion_matrix(labels_all_INI, predictions_INI_all, classes_names, title)
            classes_names = ['Left', 'Right']
            title = 'Binary'
            labels_all = np.array(labels_all)
            labels_all_LR_filter = labels_all != 1
            labels_all_LR = labels_all[labels_all_LR_filter]
            predictions_LR_all = np.array(predictions_LR_all).flatten()
            predictions_LR_all = predictions_LR_all[labels_all_LR_filter]
            plot_confusion_matrix(labels_all_LR, predictions_LR_all, classes_names, title)
            plt.show()
    else:
        print("Unsupported action: {}\nSupported actions are: 'train', 'predict', 'committee_predict' and 'test_performance'".format(action))




