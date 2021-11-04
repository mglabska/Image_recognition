# Imports
import os
import PIL
import splitfolders
from keras.preprocessing.image import ImageDataGenerator


# Split folders into training, validation and test set (with splitfolders library, to be installed first)
class Folders:
    def __init__(self, given_path, target_path):
        self.given_path = given_path
        if not os.path.exists(target_path):
            splitfolders.ratio(self.given_path, output=target_path, seed=1337, ratio=(.8, 0.1, 0.1))
        self.target_path = target_path
        self.train_path = self.target_path + '\\train'
        self.valid_path = self.target_path + '\\val'
        self.test_path = self.target_path + '\\test'


# Data generation
class Generator(Folders):
    def __init__(self, image_size, batch_size, given_path, target_path):
        super().__init__(given_path, target_path)
        self.img_gen = ImageDataGenerator(rescale=1. / 255)
        self.train_gen = self.img_gen.flow_from_directory(
            self.train_path, target_size=image_size,
            class_mode='binary',
            shuffle=True,
            batch_size=batch_size
        )
        self.val_gen = self.img_gen.flow_from_directory(
            self.valid_path,
            target_size=image_size,
            class_mode='binary',
            shuffle=True,
            batch_size=batch_size
        )
        self.test_gen = self.img_gen.flow_from_directory(
            self.test_path,
            target_size=image_size,
            class_mode='binary',
            shuffle=True,
            batch_size=batch_size
        )

# Preparing test images and test labels for evaluation and metrics
    def generate_test_images(self):
        test_images = []
        for i in range(self.test_gen.__len__()):
            test_images.extend(
                self.test_gen.__getitem__(i)[0]
            )
        return test_images

    def generate_test_labels(self):
        test_labels = []
        for j in range(self.test_gen.__len__()):
            test_labels.extend(
                self.test_gen.__getitem__(j)[1]
            )
        return test_labels


# Skip invalid data
def my_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except PIL.UnidentifiedImageError:
            pass
