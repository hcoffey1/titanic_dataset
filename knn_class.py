#Hayden Coffey
#COSC 425, Project 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Return confusion matrix
def conf_matrix(predictions, answers):
    conf = [0,0,0,0]
    for i in range(len(predictions)):
        if predictions[i] == answers[i] and predictions[i] == 0:
            conf[0]+=1

        if predictions[i] != answers[i] and predictions[i] == 1:
            conf[1]+=1

        if predictions[i] != answers[i] and predictions[i] == 0:
            conf[2]+=1

        if predictions[i] == answers[i] and predictions[i] == 1:
            conf[3]+=1

    return conf

#Calculate and print stats from confusion matrix
def print_stats(conf):
    TN = conf[0]
    FP = conf[1]
    FN = conf[2]
    TP = conf[3]
    print("Confusion Matrix:")
    print(TN, FP)
    print(FN, TP)
    TPR = "Nil"
    TNR = "Nil"
    PPV = "Nil"
    F1 = "Nil"

    accuracy = (TN + TP) / (TN + TP + FN + FP)
    if TP + FN > 0:
        TPR = TP / (TP + FN)
    if TP + FP > 0:
        PPV = TP / (TP + FP)
    if TN + FP > 0:
        TNR = TN / (TN + FP)

    if TPR != "Nil" and TNR != "Nil" and PPV != "Nil":
        F1 = 2*PPV*TPR/(PPV+TPR)

    print("Accuracy:",accuracy)
    print("TPR:", TPR)
    print("PPV:", PPV)
    print("TNR:", TNR)
    print("F1:", F1)

    return accuracy

#Returns list of k closest neighbors to specified x index as index values
def knn_list(x, matrix, k):
    kn = []
    dis = 0

    for i in range(k):
        min_index = 0
        min = 100000
        for j in range(len(matrix)):
            if j == x or j in kn:
                continue

            dis = np.linalg.norm(matrix[x]-matrix[j])

            if dis < min:
                min = dis
                min_index = j
        kn.append(min_index)
    return kn

#Form predictions from knn and same data set
def knn_class(matrix, clss, k):
    predictions = []

    for i in range(len(matrix)):
        c1 = 0
        c2 = 0
        knn = knn_list(i, matrix, k)

        for j in knn:

            if clss[j] == 0:
                c1+=1
            else:
                c2+=1

        if c1 > c2:
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions

#Form predictions and evaluate them when data sets differ (Validation sets)
def knn_val(train_data, train_clss, val_data, val_clss, k):
    predictions = []
    for i in range(len(val_data)):
        knn = knn_list(len(train_data), np.vstack([train_data, val_data[i]]), k)
        c1 = 0
        c2 = 0

        for j in knn:
            if train_clss[j] == 0:
                c1+=1
            else:
                c2+=1

        if c1 > c2:
            predictions.append(0)
        else:
            predictions.append(1)

    return print_stats(conf_matrix(predictions, val_clss))

#Returns best k value for knn, uses cross validation accuracy as metric
def eval_k(train_data, train_clss, val_data, val_clss):
    max_A = 0
    max_k = 0

    for k in range(2,9):
        print("\nTraining Set k =",k)
        predictions = knn_class(train_data, train_clss, k)
        print_stats(conf_matrix(predictions, train_clss))

        print("\nValidation Set k =",k)
        result = knn_val(train_data, train_clss, val_data, val_clss, k) 
        if result > max_A:
            max_A = result
            max_k = k
    k = 17
    for i in range(2):
        print("\nTraining Set k =",k)
        predictions = knn_class(train_data, train_clss, k)
        print_stats(conf_matrix(predictions, train_clss))

        print("\nValidation Set k =",k)
        result = knn_val(train_data, train_clss, val_data, val_clss, k) 
        if result > max_A:
            max_A = result
            max_k = k
        k = k*2 -1

    print("\nBest:",max_k,max_A)
    return max_k

if __name__ == "__main__":
    df = pd.read_csv("train_clean.csv")

    answers = df["Survived"]
    data_set = df.drop("Survived", axis=1)
    matrix = data_set.values

    best_k = 0
    best_score = 0
    for k in range(2, 15):
        predictions = knn_class(matrix, answers, k)
        tmp = print_stats(conf_matrix(predictions, answers))
        if tmp > best_score:
            best_score = tmp
            best_k = k

    print("Best k:", best_k)
    print("Best score:", best_score)
