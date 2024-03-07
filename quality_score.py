from keras.models import load_model
from util.predictions import predict_image_quality
from keras import backend as K


def custom_accuracy(y_true, y_pred, threshold=0.1):
    return K.mean(K.cast(K.abs(y_true - y_pred) < threshold, 'float32'))


def measure(img_path):
    model = load_model('../../model256_100epochs/vgg16_model.h5',
                       custom_objects={'custom_accuracy': custom_accuracy})

    predicted_quality_score = predict_image_quality(model, img_path)

    return predicted_quality_score


def main():
    image_path = "data\\512x384\\826373.jpg"
    predicted_quality_score = measure(image_path)
    print(f'Predicted Quality Score: {predicted_quality_score:.4f}')


if __name__ == "__main__":
    main()
