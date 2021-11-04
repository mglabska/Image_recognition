from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Model, Sequential


# Convolutional neural network algorithm with parameters adjusted
class Cnn:
    def __init__(self):
        self.model = Sequential()

    def build_cnn(self):

        self.model.add(Conv2D(
            filters=128,
            kernel_size=(3, 3),
            input_shape=(100, 100, 3),
            activation='relu'
            )
        )
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(
            filters=56,
            kernel_size=(3, 3),
            input_shape=(100, 100, 3),
            activation='relu'
            )
        )
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(
            filters=28,
            kernel_size=(3, 3),
            input_shape=(100, 100, 3),
            activation='relu'
            )
        )       
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
        print(self.model.summary())

    def fit_cnn(self, train_generator, val_generator, batch_size):
        self.model.fit(
                  train_generator,
                  steps_per_epoch=20000 // batch_size,
                  validation_data=val_generator,
                  validation_steps=2500 // batch_size,
                  epochs=5
        )

    def save_cnn(self, name):
        self.model.save(name)


# VGG algorithm with parameters adjusted
class VGG:
    def __init__(self):
        vgg = VGG16(input_shape=(100, 100, 3), weights='imagenet', include_top=False)
        for layer in vgg.layers:
            layer.trainable = False
        x = Flatten()(vgg.output)
        prediction = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=vgg.input, outputs=prediction)

    def build_vgg(self):
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
        print(self.model.summary())

    def fit_vgg(self, train_generator, val_generator, batch_size):
        self.model.fit(
            train_generator,
            validation_data=val_generator,
            steps_per_epoch=20000 // batch_size,
            validation_steps=2500 // batch_size,
            verbose=1,
            epochs=3
        )

    def save_vgg(self, name):
        self.model.save(name)
