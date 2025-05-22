# AUC ROC for significantly different vs. similar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

dataset = 'LIVEitW'
subset = 'images'

df_pairs = pd.read_csv(f'pairs_{dataset}_{subset}.csv')
labels = df_pairs['group']

datasets = ['KonIQ-10K', 'LIVE2+KonIQ-10K']
models = ['vgg16', 'resnet50', 'inception_v3', 'nasnet_mobile', 'efficientnet_v2_s']

for train_set in datasets:
    for model in models:
        df_scores = pd.read_csv(
            f'trained_on_{train_set}/{model}_{dataset}_{subset}.csv')

        pred_dict = dict(zip(df_scores['image_name'], df_scores['pred_MOS']))
        pair_diffs = []

        for _, row in df_pairs.iterrows():
            i = row['image_i']
            j = row['image_j']

            if i not in pred_dict or j not in pred_dict:
                continue

            pair_diffs.append(abs(pred_dict[i] - pred_dict[j]))

        # ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(labels, pair_diffs)
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
        plt.savefig(f'trained_on_{train_set}/{dataset}_{subset}/{model}_AUC_ROC_DS.png', format='png', dpi=300)
        plt.show()

        # Histogram of |Δ_model| with class coloring
        pair_diffs = np.array(pair_diffs)
        pair_labels = np.array(labels)
        bins = np.linspace(0, max(pair_diffs), 40)

        plt.figure(figsize=(7, 5))

        plt.hist(pair_diffs[pair_labels == 1], bins=bins, alpha=0.7, color='blue', edgecolor='black', label='Different')
        plt.hist(pair_diffs[pair_labels == 0], bins=bins, alpha=0.7, color='red', edgecolor='black', label='Similar')

        # THRESHOLD for FPR ≤ 5%
        thr_fpr05 = None
        for f, thr in zip(fpr, thresholds):
            if f <= 0.05:
                thr_fpr05 = thr
            else:
                break

        if thr_fpr05 is not None:
            print(f'THRESHOLD for FPR ≤ 5%: {thr_fpr05:.4f}')
            plt.axvline(thr_fpr05, color='red', linestyle='--', linewidth=2, label=f'THR \n({thr_fpr05:.4f})')

        plt.xlabel(r'$|\Delta_{model}|$', fontsize=20)
        plt.ylabel('Number of pairs', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f'trained_on_{train_set}/{dataset}_{subset}/{model}_Histogram_DS.png', format='png', dpi=300)
        plt.show()
