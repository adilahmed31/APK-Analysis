import os
from glob import glob
import json
import subprocess
import yaml
import copy
import csv
import traceback
from threading import Timer

keywords = ["contact", "call", "record", "admin", "accessibility", "remote", "mail",
            "sms", "notifications", "messages", "whatsapp", "facebook"]

def get_permission_data(apkfile):
    cmd = "ninjadroid --json %s" % apkfile
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    timer = Timer(5, output.kill)
    try:
        timer.start()
        stdout, stderr = output.communicate()
        if output.returncode == 0:
            return json.loads(stdout.decode('utf-8'))
    finally:
        timer.cancel()
    return {}

def save_code_data(apkfile):
    apkfile = str(os.path.abspath(apkfile))
    outfile = apkfile.rsplit('/',1)[-1].rsplit('.',1)[0] + ".json"
    #outdir = apkfile.rsplit('/',2)[0] + "/output/"
    outdir = "/Users/adil/Documents/IPV/APKs/output/"
    fileName = outdir + outfile
    if os.path.exists(fileName):
        print(fileName + " already exists. Skipping. \n")
        return
    #TODO: link to droidlysis fork
    cmd = "droidlysis -o /tmp -i " + apkfile + " > " + outdir + outfile #droidlysis on the local system has been patched to generate json output in this instance
    os.system(cmd)
    print("Created JSON metadata file for " + apkfile + " at " + outdir + outfile + "\n")
    #output = subprocess.Popen(args_str,shell=True, stdout=subprocess.PIPE)
    #timer = Timer(10, output.kill)

def get_code_data(apkfile):
    apkfile = str(os.path.abspath(apkfile))
    inputfile = apkfile.rsplit('/',2)[0] + "/output/" + apkfile.rsplit('/',1)[-1].rsplit('.',1)[0] + ".json"
    with open(inputfile, 'r') as file:
        data = json.loads(file.read())
    return data


def get_apk_lists(directory):
    result = [y for x in os.walk(directory) for y in glob(os.path.join(x[0], '*.apk'))]
    return result

def get_output_data_2022(app):
    inputfile = "/Users/adil/Documents/IPV/APKs/output-2022/" + app + ".json"
    with open(inputfile, 'r') as file:
        data = json.loads(file.read())
    return data

def generate_json_output(apkFiles):
    for apk in apkFiles:
        print("Parsing " + apk)
        save_code_data(apk)

def generate_yaml(apkFiles,outputFile):
    data = dict()
    outdir = outputFile
    for apk in apkFiles:
        try:  
            #result=get_code_data(apk)
            result = get_output_data_2022(apk)
            package = result["manifest_properties"]["package_name"]
            permission = result["manifest_properties"]["permissions"]
            providers = [x.strip('\'') for x in result["manifest_properties"]["providers"]]
            services = [x.strip('\'') for x in result["manifest_properties"]["services"]]
            receivers = [x.strip('\'') for x in result["manifest_properties"]["receivers"]]
            code_properties = [x for x in result["smali_properties"].keys() if result["smali_properties"].get(x) == True]
            manifest_properties = [x for x in result["manifest_properties"].keys() if result["manifest_properties"].get(x) == True]
            arm_properties = [x for x in result["arm_properties"].keys() if result["arm_properties"].get(x) == True]
            wide_properties = [x for x in result["wide_properties"].keys() if result["wide_properties"].get(x) == True]
            urls = result["wide_properties"]["urls"]
            data[package] = dict()
            data[package]["permissions"] = permission   
            data[package]["providers"] = providers
            data[package]["services"] = services
            data[package]["receivers"] = receivers
            data[package]["code_properties"] = code_properties
            data[package]["urls"] = urls
            data[package]["manifest_properties"] = manifest_properties
            data[package]["arm_properties"] = arm_properties
            data[package]["wide_properties"] = wide_properties
            # if "top100" in apk or "benign" in apk:
            #     data[package]["class"] = "benign"
            # elif store360 and "store360" in apk:
            #     data[package]["class"] = "unknown"
            # elif apkfab and "apkfab" in apk:
            #     data[package]["class"] = "unknown"
            # elif "tencent" in apk and tencent:
            if "ipv" in apk:
                data[package]["class"] = "ipv"
            elif "benign" in apk:
                data[package]["class"] = "benign"
            else: 
                data[package]["class"] = "unknown"
            data[package]["apk"] = apk   
        except Exception as e:
            #traceback.print_exc()
            print("Exception")
            open("errors.txt",'a').write("\n" + apk)
            print(str(e))
            continue

    with open(outputFile, 'w') as file:
        documents = yaml.dump(data, file)



    # print(len(result))
    # apk_files = []
#apk_files = get_apk_lists("/Users/adil/Documents/IPV/APKs/multilingual/2022/")
appsfile_2022 = "/Users/adil/Documents/IPV/Multilingual/all_apps_2022.csv"
apps = list()
with open(appsfile_2022,mode='r',encoding='utf-8') as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            apps.append(row["appId"])


#generate_json_output(apk_files)
generate_yaml(apps,"apks_multilingual_2022.yaml")
