from util.preprocess_datasets.white_fill_images import white_fill_images

input_folder = '../../data/datasets/FLIVE/images'
output_folder = '../../data/datasets/FLIVE-white_fill=384x512'
target_size = (384, 512)

white_fill_images(input_folder, output_folder, target_size)
