# Run some predictions on pictures from the test set

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Generate data

path = 'C:\\Users\\admin\\SDA\\ComputerVision\\Projekt\\microsoft-catsvsdogs-dataset\\predictions\\samples\\test'
models = [
    'C:\\Users\\admin\\SDA\\ComputerVision\\Projekt\\microsoft-catsvsdogs-dataset\\models\\model_cnn',
    'C:\\Users\\admin\\SDA\\ComputerVision\\Projekt\\microsoft-catsvsdogs-dataset\\models\\model_vgg'
          ]


def generate_data(img_path):
    img = ImageDataGenerator(rescale=1. / 255).flow_from_directory(img_path, target_size=[100, 100])
    return img


def display_image(index, img):
    test_images = []
    for i in range(img.__len__()):
        test_images.extend(
            img.__getitem__(i)[0]
        )
    plt.imshow(test_images[index])
    plt.show()


if __name__ == '__main__':
    image = generate_data(path)
    print(image.class_indices)
    for model in models:
        new_model = load_model(model)
        predict = new_model.predict(image)
        predictions = np.round(predict).astype(int)
        prediction_list = pd.DataFrame(data=predictions, columns=['Predictions'], dtype='float')
        prediction_list.to_csv(f'{model[-3:]}.csv', sep=',', index=False)
    display_image(0, image)
    display_image(1, image)
