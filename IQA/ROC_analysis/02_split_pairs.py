# 2nd step - split groups
import pandas as pd

dataset = 'KonIQ-10K'
subset = 'test'

df = pd.read_csv(f'pairs_{dataset}_{subset}.csv')

df_significant = df[df['group'] == 1].drop(columns=['group'])
df_similar = df[df['group'] == 0].drop(columns=['group'])

df_significant.to_csv(f'significantly_different_pairs_{dataset}_{subset}.csv', index=False)
df_similar.to_csv(f'similar_pairs_{dataset}_{subset}.csv', index=False)
