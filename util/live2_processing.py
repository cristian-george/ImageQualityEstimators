import pandas as pd

# Read the original CSV file
df = pd.read_csv('../data/LIVE2/LIVE_image_scores.csv')

# Subtract 5 from the result of the division DMOS_score by 25
df['DMOS'] = 5 - df['DMOS_score'] / 25

# Drop the original DMOS_score column
df.drop(columns=['DMOS_score'], inplace=True)

# Save the modified dataframe to a new CSV file
df.to_csv('../data/LIVE2/LIVE2_scores.csv', index=False)
