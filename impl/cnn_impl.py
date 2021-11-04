import data_gen as dg
from algorithms_class import Cnn
from data_prep_class import my_gen

if __name__ == '__main__':
    gen = dg.generate_data()
    test_labels = dg.generate_test_labels()
    test_images = dg.generate_test_images()
    model_cnn = Cnn()
    model_cnn.build_cnn()
    train_gen = my_gen(gen.train_gen)
    val_gen = my_gen(gen.val_gen)
    test_gen = my_gen(gen.test_gen)
    model_cnn.fit_cnn(train_gen, val_gen, dg.batch_size)
    model_cnn.save_cnn(
        'C:\\Users\\admin\\SDA\\ComputerVision\\Projekt\\microsoft-catsvsdogs-dataset\\models\\model_cnn'
    )
