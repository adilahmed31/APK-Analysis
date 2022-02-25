from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import json
import csv
import pandas as pd
from decision_tree import *
# Reference: https://scikit-learn.org/stable/modules/feature_selection.html

def get_stats(features, data1, data2, data1_features, data2_features, label="label"):

    data1_ipv_rows = data1.loc[data1[label] == 1]
    data2_ipv_rows = data2.loc[data2[label] == 1]
    length1 = data1_ipv_rows.shape[0]
    length2 = data2_ipv_rows.shape[0]

    stats = dict()
    for feature in features:
        if feature in data1_features and feature in data2_features:
            stats[feature] = dict()
            feature_total_data1 = (float)(data1[feature].sum())
            feature_ipv_data1 = (float)(data1_ipv_rows[feature].sum())
            feature_benign_data1 = feature_total_data1 - feature_ipv_data1

            feature_total_data2 = (float)(data2[feature].sum())
            feature_ipv_data2 = (float)(data2_ipv_rows[feature].sum())
            feature_benign_data2 = feature_total_data2 - feature_ipv_data2



            stats[feature]["Data1"] = feature_ipv_data1 / length1
            stats[feature]["Data2"] = feature_ipv_data2 / length2
            if stats[feature]["Data1"] == 0:
                stats[feature]["Percent"] = 0
            else:
                stats[feature]["Percent"] = round(((stats[feature]["Data2"] - stats[feature]["Data1"]) 
                                                / stats[feature]["Data1"]) * 100, 2)
    return stats

def read_data(inputcsv):
    data = pd.read_csv(inputcsv)
    # del train_data["app_id"]
    label = data.label # Target variable
    # del data["label"]
    app_id = data.app_id
    del data["app_id"]
    columns = data.columns
    return data, label, columns, app_id

def rem_features_low_varience(X, threshold=0.8, rfe=True):
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    # print("Before: ", X.shape)
    trimmed = sel.fit_transform(X)
    trimmed_cols = X.columns[(sel.get_support())]
    # print("After: ", trimmed.shape)
    # print(trimmed_cols)
    return X[trimmed_cols], trimmed_cols

def rec_feature_selection(X, y):
    svc = SVC(kernel="linear")
    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy',
              min_features_to_select=min_features_to_select)
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select,
                len(rfecv.grid_scores_) + min_features_to_select),
            rfecv.grid_scores_)
    plt.show()


def main():
    reverse = False
    trim = True
    train_file = "apk_train_data.csv"
    X_train, Y_train, train_columns, app_id = read_data(train_file)
    # test_file = "apk_test_data.csv"
    test_file = "playstore.csv"
    X_test, Y_test, test_columns, _ = read_data(test_file)
    # Trying Recursive feature elimination with cross-validation
    # Takes time to run, commenting out
    # rec_feature_selection(X_train, Y_train)

    # Trimmed based on low varience
    trimmed_train, trimmed_train_cols = rem_features_low_varience(X_train)
    trimmed_test, trimmed_test_cols = rem_features_low_varience(X_test)
    print(json.dumps(get_stats(trimmed_train_cols, trimmed_train, X_test, trimmed_train_cols, test_columns), sort_keys=False, indent=4))

    print(json.dumps(get_stats(trimmed_test_cols, trimmed_test, X_train, trimmed_test_cols, train_columns), sort_keys=False, indent=4))


    # Testing the trimmed data
    # models, _ = decision_tree(trimmed_train, Y_train, cvol=False)
    # acc_score = evaluate_pred(models, X_test, trimmed_train_cols, test_columns, Y_test=Y_test)

    if reverse and not trim:
        # Reversing the test/train data, but data is not trimmed
        print("Reversing the test/train data, but data is not trimmed")
        models, _ = decision_tree(X_test, Y_test, cvol=False)
        acc_score = evaluate_pred(models, X_train, test_columns, train_columns, Y_test=Y_train)
    
    if reverse and trim:
        # Reversing the test/train data and data is trimmed
        print("Reversing the test/train data and data is trimmed")
        models, _ = decision_tree(trimmed_test, Y_test, cvol=False)
        acc_score = evaluate_pred(models, X_train, trimmed_test_cols, train_columns, Y_test=Y_train)


if __name__ == '__main__':
    #main()
    train_file = "apk_dataset_consolidated_train.csv"
    X_train, Y_train, train_columns, app_id = read_data(train_file)
    # test_file = "apk_test_data.csv"
    test_file = "apk_dataset_consolidated_test.csv"
    X_test, Y_test, test_columns, _ = read_data(test_file)
    stats = get_stats(train_columns, X_train, X_test, train_columns, test_columns)
    fieldnames = stats.keys()
    # with open('ipv_frequency_analysis.csv', 'w') as f:
    #     for key in fieldnames:
    #         f.write("%s,%s,%s,%s\n"%(key,stats.get(key).get("Data1"),
    #             stats.get(key).get("Data2"),stats.get(key).get("Percent")))