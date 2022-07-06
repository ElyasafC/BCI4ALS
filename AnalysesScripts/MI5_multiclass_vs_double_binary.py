'''
This experiment compares the standard multiclass classifiers to the
two binary classifier architecture for classifying motor imagery bci signals.
'''


import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import scipy.io
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from pathlib import Path
import time

labels_to_names = {1: "idle",
                   2: "left",
                   3: "right",
                   4: "left_or_right",
                   -1: "none"}

def load_all_features_data(directory_path):
    '''
    loads the features data
    :param directory_path: directory path
    :return: np array of side n*d for n trials with features dimension d.
    '''
    all_features =  np.array(scipy.io.loadmat(directory_path + "\\AllDataInFeatures.mat")["AllDataInFeatures"])
    try:
        all_labels = np.array(scipy.io.loadmat(directory_path + "\\AllDataLabels.mat")["AllDataLabels"][0])
    except:
        try:
            all_labels = np.array(scipy.io.loadmat(directory_path + "\\trainingVec.mat")["trainingVec"][0])
        except:
            try:
                all_labels = np.array(scipy.io.loadmat(directory_path + "\\trainingVec.mat")["trainingVecBeg"][0])
            except:
                raise Exception("didnt find labels!")
    return all_features, all_labels

def flatten(t):
    '''
    returns a flattened copy of a list of lists
    '''
    return [item for sublist in t for item in sublist]

def train_all():
    '''
    trains multiple models and compares the preformence between using one multiclass
    model to using two binary classifiers:
        one to identipy idle/not idle
        one to identify the non idle between left and right.
    :return: None. the function prints graphs to the screen and saves them to disk.
    '''
    datasets_paths_prefix = r"C:\Users\Elyasaf\Documents\school\BCI\\" # change to path on computer
    graphs_path_prefix = datasets_paths_prefix + "BCI4ALS\\graphs\\"
    #we found the easiest way to edit this is to comment in and out whatever dataset is being compared.
    datasets_paths = [
#                     r"recordings\1.5\first\chosen electrodes(without 3)",
                    r"recordings\1.5\first\electrodes 1-11",
                    r"recordings\1.5\second\electrodes 1-11",
#                     r"recordings\1.5\third\chosen electrodes",
                    r"recordings\1.5\third\electrodes 1-11",
#                     r"recordings\27.3\first\chosen electrodes_ 1-4, 6-11",
                    r"recordings\27.3\first\electrodes 1-11",
                    r"recordings\11.4\first\electrodes 1-11",
                    r"recordings\11.4\second\electrodes 1-11",
#                     r"recordings\11.4\third\chosen electrodes_ 1-9, 11",
                    r"recordings\11.4\third\electrodes 1-11",
#                     r"recordings\24.4\first\chosen electrodes (without 4,5,11)",
                    r"recordings\24.4\first\electrodes 1-11",
#                     r"recordings\24.4\second\chosen electrodes (without 5,11)",
                    r"recordings\24.4\second\electrodes 1-11",
#                     r"recordings\24.4\third\chosen electrodes (without 1,4,5,11)",
                    r"recordings\24.4\third\electrodes 1-11",
                    r"recordings\22.5\third\electrodes 1-11",
    ]
    # initialize accumulators
    n_datasets = len(datasets_paths)
    X_all = [None] * n_datasets
    Y_all = [None] * n_datasets
    DBC_improvements_list = []
    DBC_complete_accuracy_list = []
    best_model_multiclass_accuracy_list = []
    models_per_iterations_INI_list = []
    models_per_iterations_LR_list = []
    for i_dataset in range(len(datasets_paths)):
        print(f"%%%%%%%%%% Datasets progression: {i_dataset}/{len(datasets_paths)} %%%%%%%%%%")
        print(f"dataset: {datasets_paths[i_dataset]}")
        X_all[i_dataset], Y_all[i_dataset] = load_all_features_data(datasets_paths_prefix + datasets_paths[i_dataset])
        # set number of experiment repetitions. the higher the number the more stability of the results.
        experiment_repetitions = 3
        models_per_iterations = []
        predictions_per_iteration = []
        y_test_per_iteration = []
        for i in range(experiment_repetitions):
            print(f"iteration: {i}/{experiment_repetitions}")
            X, y = X_all[i_dataset], Y_all[i_dataset].reshape(X_all[i_dataset].shape[0], 1)
#             X = X[:, 9:]  # Remove csp features if they are included
            data = np.concatenate((X, y), axis=1)

            # balance dataset by each class.
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

            # shuffle data to prevent lazypredict 'DummyClassifier' from learning data order
            np.random.shuffle(training)
            np.random.shuffle(test)

            # split train and test
            X_train, y_train = training[:, :-1], training[:, -1]
            X_test, y_test = test[:, :-1], test[:, -1]

            # select top fetures
            selector = SelectKBest(score_func=f_classif, k=10)
            fit = selector.fit(X_train, y_train)
            X_train = fit.transform(X_train)
            X_test = fit.transform(X_test)

            # run 27 lazypredict catagorical classifiers
            clf = LazyClassifier(predictions=True)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            best_model_multiclass_accuracy = float(models.sort_values('Accuracy').iloc[-1]['Accuracy'])

            models_per_iterations.append(models)
            predictions_per_iteration.append(predictions)
            y_test_per_iteration.append(y_test)


            ## idle not idle classifier:
            y_train_INI = y_train.copy()
            y_train_INI[np.logical_or(y_train_INI == 2, y_train_INI == 3)] = 4
            y_test_INI = y_test.copy()
            y_test_INI[np.logical_or(y_test_INI == 2, y_test_INI == 3)] = 4

            selector_INI = SelectKBest(score_func=f_classif, k=10)
            fit_INI = selector_INI.fit(X_train, y_train_INI)
            X_train_INI = fit_INI.transform(X_train)
            X_test_INI = fit_INI.transform(X_test)


            clf_INI = LazyClassifier(predictions=True)
            models_INI, predictions_INI = clf_INI.fit(X_train_INI, X_test_INI, y_train_INI, y_test_INI)
            best_model_INI = models_INI.sort_values('Accuracy').iloc[-1]

            ## left right classifier:
            train_LR_filter = y_train != 1
            y_train_LR, x_train_LR = y_train[train_LR_filter], X_train[train_LR_filter]
            test_LR_filter = y_test != 1
            y_test_LR, X_test_LR = y_test[test_LR_filter], X_test[test_LR_filter]

            selector_LR = SelectKBest(score_func=f_classif, k=10)
            fit_LR = selector_LR.fit(x_train_LR, y_train_LR)
            X_train_LR = fit_LR.transform(x_train_LR)
            X_test_LR = fit_LR.transform(X_test_LR)

            clf_LR = LazyClassifier(predictions=True)
            models_LR, predictions_LR = clf_LR.fit(X_train_LR, X_test_LR, y_train_LR, y_test_LR)

            best_model_LR = models_LR.sort_values('Accuracy').iloc[-1]

            best_model_INI_preds_on_LR = predictions_INI[best_model_INI.name].iloc[test_LR_filter]
            best_model_LR_preds = predictions_LR[best_model_LR.name]
            correct_INI_in_LR = best_model_INI_preds_on_LR == 4
            correct_LR = best_model_LR_preds == y_test_LR

            #DoubleBinaryClassifier preformance analysis
            DBC_correct_on_LR = np.logical_and(correct_INI_in_LR.values, correct_LR.values)
            DBC_accuracy_ON_LR = sum(DBC_correct_on_LR)/DBC_correct_on_LR.size

            test_INI_filter = y_test == 1  # ( = ~test_LR_filter)
            predictions_on_idle = predictions_INI[best_model_INI.name].iloc[test_INI_filter]
            DBC_correct_on_idle = predictions_on_idle == 1
            DBC_accuracy_on_idle = sum(DBC_correct_on_idle)/DBC_correct_on_idle.size
            DBC_complete_accuracy = (sum(DBC_correct_on_LR) + sum(DBC_correct_on_idle)) / (DBC_correct_on_LR.size + DBC_correct_on_idle.size)
            DBC_complete_accuracy_list.append(DBC_complete_accuracy)
            DBC_improvement = DBC_complete_accuracy - best_model_multiclass_accuracy
            DBC_improvements_list.append(DBC_improvement)
            print(f"DBC accuracy on left right: {DBC_accuracy_ON_LR:.3f}, DBC accuracy on idle: {DBC_accuracy_on_idle:.3f}")
            print(f"Double Binary accuracy: {DBC_complete_accuracy:.3f}, Multiclass accuracy: {best_model_multiclass_accuracy:.3f} improvement: {DBC_improvement:.3f}")

            models_per_iterations.append(models)
            predictions_per_iteration.append(predictions)
            y_test_per_iteration.append(y_test)
            models_per_iterations_INI_list.append(models_INI)
            models_per_iterations_LR_list.append(models_LR)
            best_model_multiclass_accuracy_list.append(best_model_multiclass_accuracy)

    models_mean_accuracy_INI = pd.concat([p['Accuracy'] for p in models_per_iterations_INI_list], axis=1).mean(axis=1).sort_values(ascending=False)
    models_mean_accuracy_LR = pd.concat([p['Accuracy'] for p in models_per_iterations_LR_list], axis=1).mean(axis=1).sort_values(ascending=False)

    dataset_graphs_path_prefix = graphs_path_prefix

    #create folder for graphs, if doesn't exist
    Path(dataset_graphs_path_prefix).mkdir(parents=True, exist_ok=True)

    #print all statistics
    print(f"statistics on {len(datasets_paths)} datasets, {experiment_repetitions} reps.")
    plt.figure(figsize=(12,9))
    title = f"Idle-not-Idle Accuracy. datasets {len(datasets_paths)} reps {experiment_repetitions}"
    plt.title(title)
    container = plt.bar(models_mean_accuracy_INI.keys()[:10], models_mean_accuracy_INI.values[:10])
    plt.bar_label(container, labels=[f"{a:.3f}" for a in models_mean_accuracy_INI.values[:10]])
    plt.xticks(rotation=12, ha="right")
    print("\nIdle-not-Idle:\n" + str(models_mean_accuracy_INI))
    plt.savefig(dataset_graphs_path_prefix + title + '.png')
    plt.show(block=False)

    plt.figure(figsize=(12, 9))
    title = f"Left-Right Accuracy datasets {len(datasets_paths)} reps {experiment_repetitions}"
    plt.title(title)
    container = plt.bar(models_mean_accuracy_LR.keys()[:10], models_mean_accuracy_LR.values[:10])
    plt.bar_label(container, labels=[f"{a:.3f}" for a in models_mean_accuracy_LR.values[:10]])
    plt.xticks(rotation=12, ha="right")
    print("\nLeft-Right:\n" + str(models_mean_accuracy_LR))
    plt.savefig(dataset_graphs_path_prefix + title + '.png')
    plt.show(block=False)

     #close each iteration
    print(f"DBC_improvements_list mean: {np.mean(DBC_improvements_list)}")
    print(f"DBC_complete_accuracy_list mean: {np.mean(DBC_complete_accuracy_list)}")
    print(f"best_model_multiclass_accuracy_list mean: {np.mean(best_model_multiclass_accuracy_list)}")
    plt.show(block=True)

if __name__ == '__main__':
    '''
    time the experiment and make sure output is printed.
    '''
    start_time = time.time()
    print("classification comparison sequence")
    train_all()
    print("finished!")
    print(f"run time: {time.time() - start_time:.0f} seconds")
    plt.show(block=True)

