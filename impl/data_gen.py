import matplotlib.pyplot as plt
import numpy as np
from data_prep_class import Generator
from glob import glob
from keras.preprocessing import image

image_size = [100, 100]
batch_size = 32
given_path = 'C:\\Users\\admin\\SDA\\ComputerVision\\Projekt\\microsoft-catsvsdogs-dataset\\data'
target_path = 'C:\\Users\\admin\\SDA\\ComputerVision\\Projekt\\microsoft-catsvsdogs-dataset\\data\\output'


def generate_data():
    gen = Generator(image_size, batch_size, given_path, target_path)
    return gen


def generate_test_images():
    return generate_data().generate_test_images()


def generate_test_labels():
    return generate_data().generate_test_labels()
    # Display a sample image, just for test


def display_image():
    image_files = glob(generate_data().target_path + '/*/*/*.jp*g')
    plt.imshow(
        image.img_to_array(
            image.load_img(
                np.random.choice(image_files)
            )
        ).astype('uint8')
    )
    plt.show()
