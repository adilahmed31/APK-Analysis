import yaml
import csv
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function

def get_columns(data):
    columns = ["app_id", "label"]
    # default_permission_provider = ["android.hardware", "android.permission", "com.google", "com.android"]
    # column = ["app_id", "hide_app", "label"]
    for item in data: 
        if "receivers" in item:
            for field in item["receivers"]:
                field = field.split(".")[-1].lower()
                col = "receivers_%s" %(field)
                if col not in columns:
                    columns.append(col) 
                    
        if "permissions" in item:
            for field in item["permissions"]:
                field = field.split(".")[-1].lower()
                # third_party = True
                # for i in default_permission_provider:
                #     if i in field:
                #         third_party = False
                # if third_party:
                #     field = field.split(".")
                #     col = "permission_%s" %(".".join(field))
                # else:
                #     field = field.split(".")[-1].lower()
                #     col = "permission_%s" %(field)
                col = "permission_%s" %(field)
                if col not in columns:
                    columns.append(col)
        
        if "services" in item:
            for field in item["services"]:
                field = field.split(".")[-1].lower()
                col = "services_%s" %(field)
                if col not in columns:
                    columns.append(col) 
        
        if "providers" in item:
            for field in item["providers"]:
                field = field.split(".")[-1].lower()
                col = "providers_%s" %(field)
                if col not in columns:
                    columns.append(col) 

        if "code_properties" in item:
            for field in item["code_properties"]:
                field = field.split(".")[-1].lower()
                col = "code_properties_%s" %(field)
                if col not in columns:
                    columns.append(col)        
        
        if "smali_properties" in item:
            for field in item["smali_properties"]:
                field = field.split(".")[-1].lower()
                col = "smali_properties_%s" %(field)
                if col not in columns:
                    columns.append(col)   
        
        if "arm_properties" in item:
            for field in item["arm_properties"]:
                field = field.split(".")[-1].lower()
                col = "arm_properties_%s" %(field)
                if col not in columns:
                    columns.append(col)   
        
        if "wide_properties" in item:
            for field in item["wide_properties"]:
                field = field.split(".")[-1].lower()
                col = "wide_properties_%s" %(field)
                if col not in columns:
                    columns.append(col)  

    return columns

def parser(data, columns):
    output_data = dict()
    for item in data: 
        appid = item["appid"]
        output_data[appid] = dict()
        for col in columns:
            output_data[appid][col] = 0

        output_data[appid]["app_id"] = appid
        # item = data[appid]
        # print(item)
        if item["class"] == "ipv":
            output_data[appid]["label"] = 1

        if "receivers" in item:
            # print(item)
            for field in item["receivers"]:
                # print(key)
                field = field.split(".")[-1].lower()
                col = "receivers_%s" %(field)
                if col in columns:
                    output_data[appid][col] = 1
                    
        if "permissions" in item:
            for field in item["permissions"]:
                field = field.split(".")[-1].lower()
                col = "permission_%s" %(field)
                if col in columns:
                    output_data[appid][col] = 1
        
        if "services" in item:
            for field in item["services"]:
                field = field.split(".")[-1].lower()
                col = "services_%s" %(field)
                if col in columns:    
                    output_data[appid][col] = 1
        
        if "providers" in item:
            for field in item["providers"]:
                field = field.split(".")[-1].lower()
                col = "providers_%s" %(field)
                if col in columns:    
                    output_data[appid][col] = 1

        if "code_properties" in item:
            for field in item["code_properties"]:
                field = field.split(".")[-1].lower()
                col = "code_properties_%s" %(field)
                if col in columns:    
                    output_data[appid][col] = 1       
        
        if "smali_properties" in item:
            for field in item["smali_properties"]:
                field = field.split(".")[-1].lower()
                col = "smali_properties_%s" %(field)
                if col in columns:    
                    output_data[appid][col] = 1  
        
        if "arm_properties" in item:
            for field in item["arm_properties"]:
                field = field.split(".")[-1].lower()
                col = "arm_properties_%s" %(field)
                if col in columns:    
                    output_data[appid][col] = 1  
        
        if "wide_properties" in item:
            for field in item["wide_properties"]:
                field = field.split(".")[-1].lower()
                col = "wide_properties_%s" %(field)
                if col in columns:    
                    output_data[appid][col] = 1 
    return output_data

def write_csv(output_file, output_data, columns):
    with open(output_file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for key in output_data:
            row = output_data[key]
            # print(row)
            writer.writerow(row)


def main():
    input_file = "apks_multilingual_2022.yaml"
    output_file = "apks_multilingual_2022_train.csv"
    test_file = "apks_multilingual_2022_test.csv"
    data = dict()
    offstore = []
    with open(input_file, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    features = []
    labels = []
    # one_offshore = False
    for item in data:
        data[item]["appid"] = item
        # if "offstore" in data[item]["apk"]: #and one_offshore: 
        #     offstore.append(data[item])
        #     continue
        # elif "offstore" in data[item]["apk"] and not one_offshore:
        #     one_offshore = True
        features.append(data[item])
        labels.append(data[item]["class"])
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1) # 80% training and 20% test

    train_columns = get_columns(X_train)
    train_data = parser(X_train, train_columns)
    print("No. of Columns in Training: %d", len(train_columns))
    write_csv(output_file, train_data, train_columns)

    test_columns = get_columns(X_test)
    test_data = parser(X_test, test_columns)
    write_csv(test_file, test_data, test_columns)
    print("No. of Columns in Testing: %d", len(test_columns))

def generate_csv(inputyaml, outputcsv):
    data = dict()
    with open(inputyaml, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    features = []
    labels = []
    
    for item in data:
        data[item]["appid"] = item
        features.append(data[item])
        labels.append(data[item]["class"])

    columns = get_columns(features)
    csv_data = parser(features, columns)
    print("No. of Columns: %d", len(columns))
    write_csv(outputcsv, csv_data, columns)

if __name__ == '__main__':
    generate_csv("apks_multilingual_2022.yaml", "apks_multilingual_2022.csv")
    #main()