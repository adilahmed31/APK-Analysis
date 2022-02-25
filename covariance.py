import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
# Reference: https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf

def read_data(inputcsv):
    data = pd.read_csv(inputcsv)
    # del train_data["app_id"]
    label = data.label # Target variable
    del data["label"]
    app_id = data.app_id
    del data["app_id"]
    columns = data.columns
    return data, label, columns, app_id

def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns

def main():
    train_file = "apks_multilingual_trial_train.csv"
    X_train, Y_train, train_columns, app_id = read_data(train_file)
    df = pd.DataFrame(X_train, columns = train_columns)
    # Generating the correlation matrix
    correlation_mat = df.corr()

    # we compare the correlation between features and remove one of 
    # two features that have a correlation higher than threshold
    columns = np.full((correlation_mat.shape[0],), True, dtype=bool)
    threshold = 0.75

    for i in range(correlation_mat.shape[0]):
        for j in range(i+1, correlation_mat.shape[0]):
            if correlation_mat.iloc[i,j] >= threshold:
                if columns[j]:
                    columns[j] = False

    selected_columns = df.columns[columns]
    data = df[selected_columns]
    # Now, the dataset has only those columns with correlation less than threshold


    print(df.shape)
    print(data.shape)
    print(Y_train.shape)

    SL = 0.05
    data_modeled, selected_columns = backwardElimination(data.values, Y_train.values, SL, selected_columns)
    '''
    This is what we are doing in the above code block:

    We assume to null hypothesis to be “The selected combination of dependent variables do not have any effect on the independent variable”.
    Then we build a small regression model and calculate the p values.
    If the p values is higher than the threshold, we discard that combination of features.
    '''
    result = pd.DataFrame()
    result['label'] = Y_train
    data = pd.DataFrame(data = data_modeled, columns = selected_columns)
    

    # fig = plt.figure(figsize = (20, 25))
    # j = 0
    # for i in data.columns:
    #     plt.subplot(6, 4, j+1)
    #     j += 1
    #     sns.distplot(data[i][result['label']==0], color='g', label = 'benign')
    #     sns.distplot(data[i][result['label']==1], color='r', label = 'malignant')
    #     plt.legend(loc='best')
    #     if j >= 24:
    #         break
    # fig.suptitle('Apk Analysis')
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.95)
    # plt.show()
    data['label'] = Y_train
    data['app_id'] = app_id
    print(data.shape)
    data.to_csv('reduced_apks_multilingual_trial_train.csv', index=False)    
    

if __name__ == '__main__':
    main()
