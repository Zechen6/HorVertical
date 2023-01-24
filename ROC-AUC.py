import os
import numpy as np
from CVSOperator import CSVOperator
import matplotlib.pyplot as plt
import threading

# paths = ["Mine/","DP/","FedAvg/","FedBN/", "FedInfo/"]
# paths = ["PureLocal/", "FedInfo/"]
# paths = ["PureLocal/"]#, "FedAvg/","FedBN/"]
paths = ["FedProx/"]
# paths = ["DP/"]
def get_roc_auc(score_list):  #[+,-,groundtruth]
    score_list.sort(reverse=True)
    roc_list = []
    roc_list.append([0,0])
    roc_auc = 0
    accuracy = 0
    
    for item in score_list:
        conf = item[0]
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # 求准确率
        if item[0] > item[1] and item[2] == 0:
            accuracy += 1
        if item[0] < item[1] and item[2] != 0:
            accuracy += 1
        # 求TP、FP、TN、FN
    
        for i in range(len(score_list)):
            if score_list[i][0] >= conf: # 判定为正类
                if score_list[i][2] == 0:
                    TP += 1
                else:
                    FP += 1
            else:  # 判定为负类
                if score_list[i][2] != 0:
                    TN += 1
                else:
                    FN += 1
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        
        if FPR == roc_list[-1][1]:
            if roc_list[-1][1] == 0:
                roc_list.append([TPR, FPR])
            continue
        else:
            roc_auc += abs((roc_list[-1][0]+TPR)*(FPR-roc_list[-1][1]))/2
        roc_list.append([TPR, FPR])
    roc_auc += abs((roc_list[-1][0]+1)*(1-roc_list[-1][1]))/2
    # roc_auc = 1 - roc_auc
    # accuracy = 1 - accuracy
    if roc_auc < 0.5:
        roc_auc = 0.5
    
    return roc_auc, accuracy

count = 1
acc_list = []
roc_list = []
for path in paths:
    print(path)
    if count == 0:
        count = 1
        dir_num = 4000
        start = 1
    else:
        dir_num = 2001
        start = 1
    accuracy_list = [0] * (dir_num-start)
    roc_auc_list = [0] * (dir_num - start)
    for i in range(start, dir_num):
        files = os.listdir(path+str(i))
        roc_auc_total = 0
        acc_total = 0
        length_total = 0
        for file in files:
            csv_reader = CSVOperator(path+str(i)+"/" + file, 'r')
            score_list = []
            c = 0
            
            for row in csv_reader.reader:
                if c == 0:
                    c = 1
                    continue
                for r in range(len(row)):
                    row[r] = row[r].replace("[", "")
                    row[r] = row[r].replace("]", "")
                
                p_score = float(row[0])
                n_score = float(row[1])
                groundtruth = float(row[2])
                # if path == "Partial/":
                if groundtruth == 1:
                    groundtruth = 0
                else:
                    groundtruth = 1
                score_list.append([p_score, n_score, groundtruth])
            csv_reader.end()
            roc_auc, acc = get_roc_auc(score_list)
            roc_auc_total += roc_auc
            acc_total += acc
            length_total += len(score_list)
        accuracy_list[i-start] = acc_total/length_total
        roc_auc_list[i-start] = roc_auc_total/len(files)
    acc_list.append(accuracy_list)
    roc_list.append(roc_auc_list)

    # paths[0] = "PureLocal"
    # paths[0] = "FedInfo"
    # paths[-1] = "FedInfo2"
X = range(0, dir_num - start)
plt.ylabel("test accuracy accAvg")
plt.ylim(0.5, 0.9)
for i in range(len(acc_list)):
    print(max(acc_list[i]), acc_list[i].index(max(acc_list[i])))
    plt.plot(X, acc_list[i])
    paths[i] = paths[i].replace("/", "")
    saver = CSVOperator("AnalysisResult/accAvg/"+paths[i]+".csv","w")
    saver.write_row([["accAvg"]])
    saver.write_row([acc_list[i]])
plt.legend(paths)

    

X = range(0, dir_num - start)
plt.ylabel("test accuracy ROC-AUC")
plt.ylim(0.5, 0.8)
for i in range(len(roc_list)):
    plt.plot(X, roc_list[i])
    saver = CSVOperator("AnalysisResult/ROC-AUC/"+paths[i]+".csv","w")
    saver.write_row(["ROC-AUC"])
    saver.write_row([roc_list[i]])
plt.legend(paths)
plt.show()

