# APK-Analysis

This is the source code repo for an independent study done under Dr. Rahul Chatterjee at UW Madison. 

## Dependancies 

The source code has the following 3rd party package dependancies:
- Numpy
- Pandas
- Matplotlib
- Sklearn
- argparse
- Six
- Seaborn

**IMP:** Install our fork of [Ninjadroid](https://github.com/akshatgit/NinjaDroid). We have made one small [change](https://github.com/akshatgit/NinjaDroid/commit/afed104fdaefb8655168ed8647382bbaa64bdba1).


## Pipeline

There are three main steps in our pipeline:
- Generate a YAML file for all the Android APKS under observation. This will contain permission, reciever and service information. 
  - For e.g. we have a list of benign apps under Playstore folder. Benign apps are under 'Playstore/benign' and IPV apps are under 'Playstore/y'.
  - Now we run generate_permission_data.py script with 'Playstore' folder argument in generate_data method. This is a hard-coded because there are many corner cases which made creating a generic script difficult. 
  - E.g. usage: 
  ```shell
  python generate_permission_data.py -i Playstore -o playstore.yaml -ostat playstore_hidden_apps_stats.yaml
   ```
- Generate CSV file using the YAML generated. This CSV file will be used for training and testing ML models.
    - E.g. usage: 
   ```shell
    python generate_csv.py -i playstore.yaml -o playstore.csv
    ```
    - Remember to pass --test flag if you want to split the data into test and train dataset. 
- Train and Test ML model. We have 3 different ML models; decision tree, logistic regression and random forest. 
    - E.g. usage: 
    ```shell
    python decision_tree.py -tr apk_train_data.csv -te apk_test_data.csv 
    ```

## Feature Engineering

We explored different techniques to reduce the feature vector length by using the following techniques:
- Correlation between features
- Null hypothesis
- Removing features with low variance

Points 1&2 remove features which are linearly dependent on each other and hence, we can just drop one of them. 
Point 3 removes features which are not mostly 0s or 1s, i.e. they have very low entropy and don't contribute much to the ML model training. 

