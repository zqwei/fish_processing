# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:02:25 2017

@author: kawashimat
"""
plt.close("all")
clear_all()


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from scipy.ndimage import rotate, shift
import imfunctions as im
from numpy.random import permutation
from scipy.ndimage.filters import gaussian_filter

from keras.datasets import cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  train_features.shape
num_classes = len(np.unique(train_labels))


pathname = r"C:\Users\kawashimat\Documents\Datasets\20x_voltron\06072017Fish1"

image_radius=14
imlist=np.load(pathname+"\\imlist.npy")[()]
classlist=np.load(pathname+"\\classlist.npy")[()]




inds1=np.where(classlist==1)[0]
inds2=np.where(classlist==2)[0]

num_images=imlist.shape[0]
imwidth=imlist.shape[2]
center=int((imwidth-1)/2)

datasets=np.zeros((num_images,1,image_radius*2+1,image_radius*2+1))
for i in range(num_images):
    tmp=im.imNormalize99(imlist[i,0,center-image_radius:center+image_radius+1,center-image_radius:center+image_radius+1].squeeze())
    datasets[i,:,:,:]=tmp[None,None,:,:]
        

rand_inds=permutation(len(classlist))

train_features=datasets[rand_inds[500:],:,:,:]
test_features=datasets[rand_inds[:500],:,:,:]

train_labels = np_utils.to_categorical(classlist[rand_inds[500:]]-1,2)
test_labels  = np_utils.to_categorical(classlist[rand_inds[:500]]-1,2)


    
# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(1,image_radius*2+1, image_radius*2+1), padding='same',activation='elu'))
	model.add(Conv2D(32, (3, 3), padding='same',activation='elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
#    
	model.add(Conv2D(64, (3, 3), padding='same',activation='elu')) 
	model.add(Conv2D(64, (3, 3), padding='same',activation='elu')) 
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
    
	model.add(Conv2D(128, (3, 3), padding='same',activation='elu')) 
	model.add(Conv2D(128, (3, 3), padding='same',activation='elu')) 
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
    
    
	model.add(Flatten())
	model.add(Dense(64, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = larger_model()
# Fit the model
#model_info=model.fit(train_features, train_labels, validation_data=(test_features, test_labels), epochs=50, batch_size=32)
datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True,shear_range=np.pi/6)
model_info=model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 64), validation_data=(test_features, test_labels),\
                               epochs=50, steps_per_epoch = int(train_features.shape[0]/64))
## Final evaluation of the model
scores = model.evaluate(test_features, test_labels, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
fpath=pathname+r'\\cell_model_32.hdf5'

if os.path.exists(fpath):
    os.remove(fpath)
model.save(fpath,overwrite=True)
