import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.model_class import IQA
from util.gpu_utils import check_gpu_support, limit_gpu_memory, increase_cpu_num_threads


def predict_scores_for_images(weights_path, image_dir, image_names):
    trainer = IQA()
    trainer.model.load_weights(weights_path)
    trainer.compile_model()

    predicted_scores = []
    for image_name in image_names:
        image_path = image_dir + image_name

        predicted_score = trainer.predict_score_for_image(image_path)  # Predict score for the image
        predicted_scores.append(predicted_score)
    return predicted_scores


if __name__ == "__main__":
    use_gpu = check_gpu_support()
    if use_gpu:
        limit_gpu_memory(memory_limit=3500)
    else:
        increase_cpu_num_threads(num_threads=os.cpu_count())

    data_directory = '../data/KonIQ-10K/'
    test_images = data_directory + 'test/all_classes/'
    test_scores = data_directory + 'test_labels.csv'

    df = pd.read_csv(test_scores)

    image_names = df['image_name'].values
    true_scores = df['MOS'].values

    weights_path = '../trained_models/resnet50_finetune_huber_crop384x256/best_model.h5'
    predicted_scores = predict_scores_for_images(weights_path, test_images, image_names)
    predicted_scores = np.array(predicted_scores)

    # Calculate absolute error for each image
    absolute_errors = np.abs(predicted_scores - true_scores)

    # Plot the results
    plt.scatter(true_scores, absolute_errors)
    plt.xlabel('True Scores')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error vs. True Scores')
    plt.show()

    # Print some example predicted and true scores along with absolute errors
    for i in range(5):  # Print the first 5 examples
        print(
            f"Image Name: {image_names[i]}, "
            f"Predicted Score: {predicted_scores[i]}, "
            f"True Score: {true_scores[i]}, "
            f"Absolute Error: {absolute_errors[i]}")
