import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier # Import Random Forest Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_roc_curve
import numpy as np
import matplotlib.pyplot as plt  


def read_data(inputcsv):
    data = pd.read_csv(inputcsv)
    # del train_data["app_id"]
    label = data.label # Target variable
    del data["label"]
    app_id = data.app_id
    del data["app_id"]
    columns = data.columns
    return data, label, columns, app_id

def evaluate_pred(models, X_test, train_columns, test_columns, appid=[], Y_test=[]):
    acc_score = []
    # print(len(train_columns))
    # print(len(test_columns))
    train_feature = train_columns.intersection(test_columns)
    dummy_columns = train_columns.difference(test_columns)
    X_test = X_test[train_feature]
    X_test = X_test.reindex(X_test.columns.union(dummy_columns, sort=False), axis=1, fill_value=0)
    X_test = X_test[train_columns]
    print("Test feature vector length: %d", len(test_columns))
    print("Intersection: %d", len(train_feature))
    print("Dummy: %d", len(dummy_columns))
    # r = X_test.index[np.isnan(X_test).any(1)]
    # print(r)
    # print(len(X_test))
    # # print(X_test.iloc[380])
    threshold = 0.3
    for model in models:
        #pred_values = model.predict(X_test)
        pred_values = model.predict_proba(X_test)
        predicted = (pred_values [:,1] >= threshold).astype('int')
        if len(Y_test) == 0:
            print("No labels to compare")
            acc_score.append(pred_values)
            continue
        tpr = recall_score(Y_test, predicted)   
        tnr = recall_score(Y_test, predicted, pos_label = 0) 
        fpr = 1 - tnr
        fnr = 1 - tpr
        
        acc = accuracy_score(Y_test , predicted)
        print(confusion_matrix(Y_test, predicted))
        acc_score.append(acc)
        p_str = "Accuracy: %f FPR: %f FNR: %f" % (acc, fpr, fnr)
        print(p_str)
        plot_roc_curve(model, X_test, Y_test)  
        plt.show()  
    return acc_score

def random_forest(X, Y, cvol=True, k=5):
    models = []
    acc_score = []
    # threshold = 0.9
    if cvol: 
        kf = KFold(n_splits=k, random_state=None)
        for train_index , test_index in kf.split(X):
            X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = Y[train_index] , Y[test_index]
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            # pred_values = model.predict_proba(X_test)
            # predicted = (pred_values [:,1] >= threshold).astype('int')
            
            acc = accuracy_score(y_test , pred_values)
            print("Training: %d", acc)
            acc_score.append(acc)
            models.append(model)
    else:
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X,Y)
        models.append(model)

    return models, acc_score

def predict(models,X_test,train_columns,test_columns,Y_test):
    train_feature = train_columns.intersection(test_columns)
    dummy_columns = train_columns.difference(test_columns)
    X_test = X_test[train_feature]
    X_test = X_test.reindex(X_test.columns.union(dummy_columns, sort=False), axis=1, fill_value=0)
    X_test = X_test[train_columns]
    print("Test feature vector length: %d", len(test_columns))
    print("Intersection: %d", len(train_feature))
    print("Dummy: %d", len(dummy_columns))
    # r = X_test.index[np.isnan(X_test).any(1)]
    # print(r)
    # print(len(X_test))
    # # print(X_test.iloc[380])
    threshold = 0.3
    for model in models:
        #pred_values = model.predict(X_test)
        pred_values = model.predict_proba(X_test)
        predicted = (pred_values [:,1] >= threshold).astype('int')
    return predicted


def train(file):
    train_file = file
    X_train, Y_train, train_columns, _ = read_data(train_file)
    models, _ = random_forest(X_train, Y_train, cvol=True)

    # save
    joblib.dump(models, "./random_forest.joblib")
    test_file = "apks_multilingual_2020_test.csv"
    X_test, Y_test, test_columns, _ = read_data(test_file)
    acc_score = evaluate_pred(models, X_test, train_columns, test_columns, Y_test=Y_test)

    return models, X_test, train_columns, test_columns, Y_test
 

if __name__ == '__main__':
    models, X_test, train_columns, test_columns, Y_test = train("reduced_apks_multilingual_2020_train.csv")
    # load
    # models = joblib.load("./random_forest.joblib")
    #predict(models, X_test, train_columns, test_columns, Y_test=Y_test)