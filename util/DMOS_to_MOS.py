import pandas as pd

'''
From [0,100] where 0 means best and 100 means worst to
     [1,  5] where 1 means worst and  5 means best
'''

scores_file = '../data/LIVE2/LIVE2_scores.csv'

# Read the original CSV file
df = pd.read_csv(scores_file)

# Subtract 5 from the result of the division by 25
df['MOS'] = 5 - df['DMOS'] / 25

# Save the modified dataframe to a new CSV file
df.to_csv(scores_file, index=False)
