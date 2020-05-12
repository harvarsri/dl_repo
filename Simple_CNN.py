#All the Imports
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import MaxPooling2D , Convolution2D
from keras.layers import Dense, Dropout, Activation ,Flatten
import numpy as np

#All the Variables
img_width = 64
img_height = 64
train_dir = 'dataset/cat_dog_dataset/training_set'
test_dir = 'dataset/cat_dog_dataset/test_set'
train_size = 1000
test_size = 100
epochs = 50
batch_size = 20

single_img_dir = 'dataset/cat_dog_dataset/single_prediction/cat_or_dog_2.jpg'

if K.image_data_format() == 'channels_first':
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

model = Sequential()

model.add(Convolution2D(32,(3,3),input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(64,(3,3),input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(64,(3,3),input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = [("accuracy")])

model.fit_generator(
        train_generator,
        steps_per_epoch= train_size // batch_size,
        epochs= epochs,
        validation_data= test_generator,
        validation_steps= test_size // batch_size)

y_pred = image.load_img(single_img_dir,target_size = (img_width,img_height))
y_pred = image.img_to_array(y_pred)
y_pred = np.expand_dims(y_pred,axis = 0)
y_pred = model.predict(y_pred)

print(y_pred)

#to find the index of the file you trained
"""
To find the index of the file you trained
Code:

"""