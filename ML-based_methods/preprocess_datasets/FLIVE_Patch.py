from white_fill_images import white_fill_images

input_folder = '../Datasets/FLIVE_Patch/images'
output_folder = '../Datasets/FLIVE_Patch-white_fill=256x256'
target_size = (256, 256)

white_fill_images(input_folder, output_folder, target_size)
