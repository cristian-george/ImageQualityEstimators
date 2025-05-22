from resize_images import resize_images

input_folder = '../Datasets/LIVEitW/images'
output_folder = '../Datasets/LIVEitW-resize=512x512'
target_size = (512, 512)

resize_images(input_folder, output_folder, target_size)
