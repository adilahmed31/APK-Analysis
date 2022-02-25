import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import graphviz
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
    print("Test feature vector length: %d", len(X_test.columns))
    print("Intersection: %d", len(train_feature))
    print("Dummy: %d", len(dummy_columns))
    # r = X_test.index[np.isnan(X_test).any(1)]
    # print(r)
    # print(len(X_test))
    # # print(X_test.iloc[380])
    for model in models:
        pred_values = model.predict(X_test)
        if len(Y_test) == 0:
            print("No labels to compare")
            acc_score.append(pred_values)
            continue
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

def decision_tree(X, Y, cvol=True, k=5):
    print(Y)
    models = []
    acc_score = []
    if cvol: 
        kf = KFold(n_splits=k, random_state=None)
        for train_index , test_index in kf.split(X):
            X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = Y[train_index] , Y[test_index]
            model = DecisionTreeClassifier(criterion="entropy", max_depth=5)
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            
            acc = accuracy_score(pred_values , y_test)
            print("Training: %d", acc)
            acc_score.append(acc)
            models.append(model)
    else:
        model = DecisionTreeClassifier(criterion="entropy", max_depth=5)
        model.fit(X,Y)
        models.append(model)

    # DOT data
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(model,feature_names=X.columns,filled=True)
    fig.savefig("decision_tree.png")
    
    return models, acc_score





def main():
    train_file = "apks_multilingual_trial_train.csv"
    #train_file = "reduced_apk_train_data.csv"
    X_train, Y_train, train_columns, _ = read_data(train_file)
    print(X_train.shape)
    print(len(train_columns))
    models, _ = decision_tree(X_train, Y_train, cvol=False)

    test_file = "apks_multilingual_trial_test.csv"
    X_test, Y_test, test_columns, _ = read_data(test_file)
    acc_score = evaluate_pred(models, X_test, train_columns, test_columns, Y_test=Y_test)

    # tencent_file = "tencent.csv"
    # X_tencent, Y_tencent, tencent_columns, tencent_app_id = read_data(tencent_file)
    # acc_score = evaluate_pred(models, X_tencent, train_columns, tencent_columns, appid=tencent_app_id, Y_test=Y_tencent)

    # playstore2_file = "playstore2.csv"
    # X_ps2, Y_ps2, ps2_columns, ps2_app_id = read_data(playstore2_file)
    # acc_score2 = evaluate_pred(models, X_ps2, train_columns, ps2_columns, appid=ps2_app_id,Y_test=Y_ps2)

    # testcorpus_file = "testcorpus_en.csv"
    # X_en, Y_en, en_columns, en_app_id = read_data(testcorpus_file)
    # acc_score2 = evaluate_pred(models, X_en, train_columns, en_columns, appid=en_app_id,Y_test=Y_en)


    # apkfab_file = "apkfab.csv"
    # X_apkfab, Y_apkfab, apkfab_columns, apkfab_app_id = read_data(apkfab_file)
    # acc_score = evaluate_pred(models, X_apkfab, train_columns, apkfab_columns, appid=apkfab_app_id)

    # for pred_value in acc_score2:
    #     file = open('testcorpus_en_decision.txt', 'w')
    #     for i in range(len(pred_value)):
    #         strr = "%f %s\n" % (pred_value[i], en_app_id[i])
    #         # print(strr)
    #         file.write(strr)
    
    # for pred_value in acc_score2:
    #     df = pd.read_csv("playstore2_decision_tree.csv")
    #     for i in range(len(pred_value)):
    #         #strr = "%f %s\n" % (pred_value[i], ps2_app_id[i])
    #         df.loc[df.appId == ps2_app_id[i],'decision_tree']= pred_value[i]
    #         #df.loc[df['appId'] == ps2_app_id[i]]['decision_tree'] == str(pred_value[i])
    #     df.to_csv("playstore2_decision_tree.csv",index=False)

if __name__ == '__main__':
    main()
    