from email import charset
import subprocess
from threading import Timer
import requests
import csv
import random
from collections import OrderedDict

def playStoreCheck(appID):
    searchURL = "https://play.google.com/store/apps/details"
    payload = {'id':appID}
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    r = requests.get(searchURL, params=payload, headers=headers)
    statusCode = r.status_code
    if statusCode == 200:
        return True
    else:
        return False

def downloadApk(appID):
    cmd = "gplaycli -d %s -f /Users/adil/Documents/IPV/APKs/multilingual " % appID
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    timer = Timer(10, output.kill)
    try:
        timer.start()
        stdout, stderr = output.communicate()
    finally:
        timer.cancel()
    return True



def downloadRandomSample(inputFile):
    count = 0
    with open(inputFile,mode='r') as inp:
        reader = csv.DictReader(inp)
        apps = [row for row in reader]
    # inputFile = "/Users/adil/Documents/IPV/Multilingual/apps2.csv"
    apps = list()
    with open(inputFile,mode='r',encoding='utf-8') as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            apps.append(row["\ufeffappId"])
    for app in apps:
        try:
            if (count >= 500):
                break
            with open('downloadedApps.txt','r') as f:
                if app in f.read():
                    continue
            f.close()
            if playStoreCheck(app):
                downloadApk(app)
                with open('downloadedApps.txt','a') as f:
                    f.write("\n" + app)
                print("downloaded + " + app)
                count += 1
            else:
                print("Did not download " + app)
                continue
        except:
            continue

downloadRandomSample("/Users/adil/Documents/IPV/Multilingual/apps_multilingual_benign.csv")
#downloadApk("com.eset.parental")
