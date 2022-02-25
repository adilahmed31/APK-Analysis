import pandas as pd
import numpy as np
from decision_tree import *
from logistic_regression import *
from random_forest import *

def read_data(inputcsv):
    data = pd.read_csv(inputcsv)
    # del train_data["app_id"]
    label = data.label # Target variable
    del data["label"]
    app_id = data.app_id
    del data["app_id"]
    columns = data.columns
    return data, label, columns, app_id

def generate_component_frequencies(data, label, columns, occurrence_threshold1, occurrence_threshold2=1, count_threshold=0):
    df = pd.DataFrame(data)
    count = df.shape[0]
    ipv_frequencies = dict()
    frequencies = list()
    label = label.tolist()
    for column in columns:
        values = [int(x) for x in df[column].values]
        count = len([x for x in values if x==1])
        if count == 0:
            ipv_frequencies[column] = 0
        else:
            ipv_frequencies[column] = round((sum(values[x] for x in range(0,len(values)) if label[x] == 1) / count),2)
        list_values = [column, ipv_frequencies[column],count]
        frequencies.append(list_values)
        for fea_frequency in frequencies:
            if (fea_frequency[1] <= occurrence_threshold1 and fea_frequency[1] >= occurrence_threshold2) or fea_frequency[2] < count_threshold:
                frequencies.remove(fea_frequency)
        #df2 = pd.DataFrame(data, columns = column_names)

    with open('dump_fre_data.csv', 'w') as f:
        for frequency_list in frequencies:
            f.write("%s,%s,%s\n"%(frequency_list[0],frequency_list[1],frequency_list[2]))
    column_names = [frequencies[index][0] for index in range(len(frequencies))]
    column_names = pd.Index(column_names)
    return df[column_names], column_names


def rem_features(frequencies, ocurrence_threshold, count_threshold):
    for fea_frequency in frequencies:
        if fea_frequency[1] < ocurrence_threshold or fea_frequency[2] < count_threshold:
            del fea_frequency
    frequencies = [feature[0] for feature in frequencies]
    return frequencies


def main():
    train_file = "apk_dataset_consolidated_train.csv"
    X_train, Y_train, train_columns, app_id = read_data(train_file)
    test_file = "apk_dataset_consolidated_test.csv"
    X_test, Y_test, test_columns, _ = read_data(test_file)
    trimmed_train, trimmed_train_cols = generate_component_frequencies(X_train, Y_train, train_columns,0.6,0.2,10)

    #models, _ = decision_tree(trimmed_train, Y_train, cvol=False)
    #models, _ = logistic_regression(trimmed_train, Y_train, cvol=False)
    models, _ = random_forest(trimmed_train, Y_train, cvol=False)
    acc_score = evaluate_pred(models, X_test, trimmed_train_cols, test_columns, Y_test=Y_test)
    test_ipv = sum(Y_test)
    test_benign = len(Y_test) - test_ipv
    print("Test IPV: "+str(test_ipv))
    print("Test Benign: "+str(test_benign))

    


if __name__ == '__main__':
    main()
