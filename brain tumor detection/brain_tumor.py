from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# or classifier.add(Conv2D(64->"no of feature dector", (3, 3), input_shape = (3, 64, 64)->"if using theano backend", activation = 'relu')-> "is for non linearity,could use sigmoid")

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(MaxPooling2D(pool_size = (2, 2)->'can choose more'))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#classifier.add(Dense(units = 128->"could choose more or less generally in power of 2", activation = 'relu')) 
classifier.add(Dense(units = 1, activation = 'sigmoid'))
#classifier.add(Dense(units = 1, activation = 'sigmoid'->is used beacuse the outbut is binary btw cat and dog.if out contains more the 2 we could have ud=sed softmax activation function))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 460,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 80)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('testing/wt2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'normal'
else:
    prediction = 'tumor found'