import os
import cv2
import pandas as pd

# Load BRISQUE model
brisque_model_path = "brisque_model_live.yml"
brisque_range_path = "brisque_range_live.yml"
brisque = cv2.quality.QualityBRISQUE()

# Directory containing images
image_dir = "../data/LIVE2/images"

# Initialize a dataframe to store the results
results_df = pd.DataFrame(columns=['image_name', 'brisque'])

# Iterate over each image in the directory
for filename in os.listdir(image_dir):
    # Read the image
    img_path = os.path.join(image_dir, filename)
    img = cv2.imread(img_path)

    # Convert to grayscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate BRISQUE score
    score = brisque.compute(img, brisque_model_path, brisque_range_path)

    if score[0] < 0.0 or score[0] > 100.0:
        print(filename)

    data = {'image_name': filename, 'brisque': 5 - score[0] / 25}
    # Store the results in the dataframe
    results_df = results_df._append(data, ignore_index=True)

# Save the dataframe to a CSV file if needed
results_df.to_csv("LIVE2_opencv_brisque.csv", index=False)
