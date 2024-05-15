import pandas as pd

'''
From [0,100] where 0 means worst and 100 means best to
     [1,  5] where 1 means worst and  5 means best
'''

scores_file = '../data/FLIVE Patch/patch_labels.csv'

# Read the original CSV file
df = pd.read_csv(scores_file)

# Subtract 5 from the result of the division by 25
df['MOS'] = 1 + df['original_MOS'] / 25

# Save the modified dataframe to a new CSV file
df.to_csv(scores_file, index=False)
