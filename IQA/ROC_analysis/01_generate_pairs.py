# 1st step - generate pairs of images in similar and significantly different groups
import pandas as pd
import itertools
from scipy.stats import norm

dataset = 'KonIQ-10K'
subset = 'test'

input_file = f'../../Datasets/{dataset}/{dataset if subset == "images" else subset}_labels.csv'
df = pd.read_csv(input_file)

results = []

for (i1, row1), (i2, row2) in itertools.combinations(df.iterrows(), 2):
    img_i, img_j = row1['image_name'], row2['image_name']

    mos_i, mos_j = row1['MOS'], row2['MOS']
    sd_i, sd_j = row1['SD'], row2['SD']
    n_i, n_j = row1['N'], row2['N']

    var_i, var_j = sd_i ** 2, sd_j ** 2
    denominator = ((var_i / n_i) + (var_j / n_j)) ** 0.5

    z = abs(mos_i - mos_j) / denominator if denominator != 0 else 0
    p = norm.cdf(z)

    # 1 - significantly different, 0 - similar
    group = 1 if p > 0.95 else 0

    results.append({
        'image_i': img_i,
        'image_j': img_j,
        'MOS_i': mos_i,
        'MOS_j': mos_j,
        'z': z,
        'p': p,
        'group': group
    })

output_df = pd.DataFrame(results)
output_df.to_csv(f'pairs_{dataset}_{subset}.csv', index=False)
