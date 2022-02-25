import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
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
    print(len(train_columns))
    train_feature = train_columns.intersection(test_columns)
    dummy_columns = train_columns.difference(test_columns)
    new = test_columns.difference(train_columns)
    # for i in new:
    #     print(i)

    X_test = X_test[train_feature]
    # print(len(X_test.columns))
    X_test = X_test.reindex(X_test.columns.union(dummy_columns, sort=False), axis=1, fill_value=0)
    X_test = X_test[train_columns]
    for model in models:
        pred_values = model.predict(X_test)
        if len(Y_test) == 0:
            print("No labels to predict")
            acc_score.append(pred_values)
            continue
        acc = accuracy_score(pred_values , Y_test)
        # for i in range(len(pred_values)):
        #     if pred_values[i] != Y_test[i]:
        #         print(appid[i])
        tpr = recall_score(Y_test, pred_values)   
        tnr = recall_score(Y_test, pred_values, pos_label = 0) 
        fpr = 1 - tnr
        fnr = 1 - tpr
        
        acc = accuracy_score(Y_test , pred_values)
        print(confusion_matrix(Y_test, pred_values))
        acc_score.append(acc)
        p_str = "Accuracy: %f FPR: %f FNR: %f" % (acc, fpr, fnr)
        print(p_str)
        plot_roc_curve(model, X_test, Y_test)  
        plt.show()  
    return acc_score

def logistic_regression(X, Y, cvol=True, k=5):
    models = []
    acc_score = []
    if cvol: 
        kf = KFold(n_splits=k, random_state=None)
        for train_index , test_index in kf.split(X):
            X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = Y[train_index] , Y[test_index]
            model = LogisticRegression(solver= 'liblinear', C=1, class_weight='balanced')
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            
            acc = accuracy_score(pred_values , y_test)
            print("Training: %d", acc)
            acc_score.append(acc)
            models.append(model)
    else:
        model = LogisticRegression(solver= 'liblinear', C=1, class_weight='balanced')
        model.fit(X,Y)
        models.append(model)
    
    return models, acc_score

def main():
    train_file = "reduced_apk_dataset_consolidated_train.csv"
    X_train, Y_train, train_columns, _ = read_data(train_file)
    models, _ = logistic_regression(X_train, Y_train, cvol=False)

    test_file = "apk_dataset_consolidated_test.csv"
    X_test, Y_test, test_columns, _ = read_data(test_file)
    acc_score = evaluate_pred(models, X_test, train_columns, test_columns, Y_test=Y_test)
    test_ipv = sum(Y_test)
    test_benign = len(Y_test) - test_ipv
    print("Test IPV: "+str(test_ipv))
    print("Test Benign: "+str(test_benign))
    
    # en_file = "testcorpus_en.csv"
    # X_en, Y_en, en_columns, _ = read_data(en_file)
    # acc_score = evaluate_pred(models, X_en, train_columns, test_columns, Y_test=Y_en)
    # en_ipv = sum(Y_ten)
    # en_benign = len(Y_en) - test_ipv
    # print("Test Corpus IPV: "+str(en_ipv))
    # print("Test Corpus Benign: "+str(en_benign))

    # tencent_file = "tencent.csv"
    # X_tencent, Y_tencent, tencent_columns, tencent_app_id = read_data(tencent_file)
    # acc_score = evaluate_pred(models, X_tencent, train_columns, tencent_columns, appid=tencent_app_id, Y_test=Y_tencent)
    # tencent_ipv = sum(Y_tencent)
    # tencent_benign = len(Y_tencent) - tencent_ipv
    # print("Tencent IPV: "+str(tencent_ipv))
    # print("Tencent Benign: "+str(tencent_benign))


    # apkfab_file = "apkfab.csv"
    # X_apkfab, Y_apkfab, apkfab_columns, apkfab_app_id = read_data(apkfab_file)
    # acc_score = evaluate_pred(models, X_apkfab, train_columns, apkfab_columns, appid=apkfab_app_id)
    # for pred_value in acc_score:
    #     file = open('apkfab_logistic.txt', 'w')
    #     for i in range(len(pred_value)):
    #         strr = "%f %s\n" % (pred_value[i], apkfab_app_id[i])
    #         # print(strr)
    #         file.write(strr)



if __name__ == '__main__':
    main()
    