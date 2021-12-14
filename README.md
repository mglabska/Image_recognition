# Image_recognition
# Description
This is a simple image classifier of cat and dog images. I suggested two solutions for this problem: CNN and VGG. 
The dataset comes from Kaggle page: https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset.

# Requirements
To run predictions only, you need to install *tensorflow* package (v2.6.0) with *keras* (v2.6.0). If you want to run files from *impl* folder as well, you need to install *split-folders* (v0.4.3) additionally. Other requirements you can find in *requirements.txt* file.

# How to run predictions
To run the predictions you need to save *models* and *predictions* folders on your disc, then open the *image_classification.py* file in *predictions* folder with Python IDE (e.g. PyCharm) and replace the *path* variable with the path your images are stored in and modify the *models* variable accordingly (update it with your local directory). Note: your images must be stored in folders, and you need to give the path to the folder, not to image samples (for example, if your folder structure is like this: *C:\\Users\\admin\\animals\\samples*, you need to enter *C:\\Users\\admin\\animals\\*, not *C:\\Users\\admin\\animals\\samples\\*.
Then you can run the script. The results will be saved in csv files (one for each model): 0 is for cats, 1 is for dogs.
In the end, you will see first two images, just for check.

If for some reason you need to run model training scripts as well, you will need to modify the paths in *data_gen.py* file and then run scripts from *cnn_impl.py* and *vgg_impl.py* files. Please remember to update the target file names and paths for saved models (the last row: *model_cnn.save_cnn* or *model_vgg.save_vgg* fuction attribute) accordingly.

# Folder structure
*task_description* - contains task descriptions plus terms of use from Kaggle

*analysis* - contains jupyter notebooks with analysis and drafts.

*data* - contains data images, both downloaded and preprocessed/splitted.

*impl* - contains classes and functions for both data preprocessing and model training and saving, plus their implementation

*models* - contains saved models

*predictions* - contains imlpementation of saved models for sample images (saved in samples subfolder).



