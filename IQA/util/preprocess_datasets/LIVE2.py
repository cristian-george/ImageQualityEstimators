import os

from util.preprocess_datasets.white_fill_images import white_fill_images

input_folder = '../../data/datasets/LIVE2_classes'
output_folder = '../../data/datasets/LIVE2_classes-white_fill=384x512'

target_size = (384, 512)

for category in os.listdir(input_folder):
    white_fill_images(os.path.join(input_folder, category),
                      os.path.join(output_folder, category), target_size)
