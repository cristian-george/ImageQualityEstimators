# AUC ROC for better vs. worse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

dataset = 'KonIQ-10K'
subset = 'test'

df_pairs = pd.read_csv(f'significantly_different_pairs_symmetric_{dataset}_{subset}.csv')

datasets = ['KonIQ-10K', 'LIVE2+KonIQ-10K']
models = ['vgg16', 'resnet50', 'inception_v3', 'nasnet_mobile', 'efficientnet_v2_s']

for train_set in datasets:
    for model in models:
        df_scores = pd.read_csv(f'trained_on_{train_set}/{model}_{dataset}_{subset}.csv')
        pred_dict = dict(zip(df_scores['image_name'], df_scores['pred_MOS']))

        true_labels = []
        pred_diffs = []

        for _, row in df_pairs.iterrows():
            i = row['image_i']
            j = row['image_j']

            if i not in pred_dict or j not in pred_dict:
                continue

            mos_i = row['MOS_i']
            mos_j = row['MOS_j']
            pred_i = pred_dict[i]
            pred_j = pred_dict[j]

            if mos_i == mos_j:
                continue

            sign_label = np.sign(mos_i - mos_j)
            model_label = np.sign(pred_i - pred_j)

            true_labels.append(1 if sign_label == model_label else 0)
            pred_diffs.append(abs(pred_i - pred_j))

        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(true_labels, pred_diffs)
        roc_auc = auc(fpr, tpr)
        print(f'AUC ROC for trained_on_{train_set}/{model}_{dataset}_{subset}: {roc_auc:.4f}')

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('FPR', fontsize=20)
        plt.ylabel('TPR', fontsize=20)
        plt.legend(loc='lower right', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'trained_on_{train_set}/{dataset}_{subset}/{model}_AUC_ROC_BW.png', dpi=300)
        plt.show()

        # Percentage of correct classification in zero, showing how many times does the model correctly recognize
        # the stimulus of higher quality
        total = len(true_labels)
        correct = sum(true_labels)
        c0 = correct / total if total > 0 else 0
        print(f'C₀: {c0:.4f}')
