import numpy as np
import os
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme('talk')
np.set_printoptions(suppress=True)

def read_data(data_dir):
    indices = np.load(f"{data_dir}/indices.npy").squeeze()
    y = np.load(f"{data_dir}/y.npy").squeeze()
    y_pred = np.load(f"{data_dir}/y_pred.npy").squeeze()

    return pd.DataFrame.from_dict({'indices': indices, 'y': y, 'y_proba': y_pred}).set_index('indices')


def fbeta(TN, FP, FN, TP, beta = 1.0):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return (1 + beta ** 2) * ((precision * recall) / ((beta ** 2 * precision) + recall))


def results(data_dir, beta = 1):
    cf_file = os.path.join(data_dir, "cf.npy")
    assert os.path.isfile(cf_file)
    cf_matrix = np.load(cf_file)
    f_score = fbeta(*cf_matrix.ravel())
    accuracy = (cf_matrix[0][0] + cf_matrix[1][1])/cf_matrix.sum()
    
    print(f"F-{beta} score: {f_score:.3f}, accuracy: {accuracy:.3f}, confusion matrix: {cf_matrix.ravel()}")


def roc_auc(data_dir, subset):
    bce_df = read_data(f"{data_dir}.bce/{subset}")
    focal_df = read_data(f"{data_dir}.focal/{subset}")
    
    bce_auc = metrics.roc_auc_score(bce_df.y, bce_df.y_proba)
    focal_auc = metrics.roc_auc_score(focal_df.y, focal_df.y_proba)
    print(f"ROC-AUC- bce: {bce_auc:.3f}, focal: {focal_auc:.3f}")
    
    plt.figure(figsize=(10, 10))
    fpr, tpr, thresholds = metrics.roc_curve(bce_df.y, bce_df.y_proba)
    plt.plot(fpr, tpr, linestyle='--', color='g', label=f"BCE Loss (AUC: {bce_auc:.3f})")
    fpr, tpr, thresholds = metrics.roc_curve(focal_df.y, focal_df.y_proba)
    plt.plot(fpr, tpr, linestyle='--', color='r', label=f"Focal Loss (AUC: {focal_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"roc_auc-{data_dir}-{subset}.png", dpi=300)
    #plt.show()


if __name__=="__main__":
    data_dir = ['70P'] #['40P', '50P', '60P', '70P', '80P']
    loss_func = ['bce', 'focal']
    subdir = ['point-one', 'point-two', 'point-three', 'point-four', 'point-five']

    for d in data_dir:
        for lf in loss_func:
            for sub in subdir:
                model_dir = os.path.join('./', f"{d}.{lf}", sub)
                print(model_dir, end=": ")
                results(model_dir)

    for d in data_dir:
        for sub in subdir:
            print(d, sub, end=": ")
            roc_auc(d, sub)

