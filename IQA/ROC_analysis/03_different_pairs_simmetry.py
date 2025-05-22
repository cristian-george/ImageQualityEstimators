# 3rd step - consider each significantly different pair in both negative and positive score differences
import pandas as pd

dataset = 'KonIQ-10K'
subset = 'test'

df = pd.read_csv(f'significantly_different_pairs_{dataset}_{subset}.csv')

df_reversed = df.rename(columns={
    'image_i': 'image_j',
    'image_j': 'image_i',
    'MOS_i': 'MOS_j',
    'MOS_j': 'MOS_i'
})[['image_i', 'image_j', 'MOS_i', 'MOS_j', 'z', 'p']]

df_symmetric = pd.concat([df, df_reversed], ignore_index=True)

df_symmetric.to_csv(f'significantly_different_pairs_symmetric_{dataset}_{subset}.csv', index=False)
