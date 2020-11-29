# building powerful image classification models using very little data
# image dataset path => C:\dogs-vs-cats

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=0.4,
    
)
