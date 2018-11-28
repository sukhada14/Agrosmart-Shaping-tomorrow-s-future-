from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import h5py


import os
from numpy import *
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from PIL import Image
from keras import backend as K
K.set_image_dim_ordering('th')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_rows=200
img_cols=200
batch_size=32

nb_classes=2
nb_epoch=10
img_channels=1
nb_filters=32
nb_pool=2
nb_conv=3


path1='D:\\ty sem 6\\wce_hackathon\\New folder\\final\\peach'
path2='D:\\ty sem 6\\wce_hackathon\\New folder\\final\\peach_resized'

print("listing all images")
listing=os.listdir(path1)
num_samples=size(listing)
print (num_samples)

print("resizing all images")

for file in listing:
	im=Image.open(path1+'\\'+file)
	img=im.resize((img_rows,img_cols))
	gray=img.convert('L')
	gray.save(path2+"\\"+file,'JPEG')


print("listing all gray images")
imlist=os.listdir(path2)



#print(imlist)


im1=array(Image.open('final\\peach_resized'+'\\'+imlist[0]))
m,n=im1.shape[0:2]

imnbr=len(imlist)
immatrix=array([array(Image.open('final\\peach_resized'+'\\'+im2)).flatten() for im2 in imlist],'f')

print("labeling all images")
label=np.ones((num_samples,),dtype=int)
label[0:2658]=0
label[2658:]=1


data,Label=shuffle(immatrix,label,random_state=2)


train_data=[data,Label]
img=immatrix[167].reshape(img_rows,img_cols)
print(img)
plt.imshow(img,cmap='gray')
print(train_data[0].shape)

print("tarin test split")

(x,y)=(train_data[0],train_data[1])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)


x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cols)

#programmers tricks for optimization
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

x_train/=255
x_test/=255

print('x_train shape',x_train.shape)
print(x_train.shape[0],'train sample')
print(x_test.shape[0],'test sample')


y_train=np_utils.to_categorical(y_train,nb_classes)
y_test=np_utils.to_categorical(y_test,nb_classes)

i=100
plt.figure()
plt.imshow(x_train[i,0],interpolation='nearest')
print('label:',y_train[i,:])


print("building model")


model=Sequential()
model.add(Convolution2D(nb_filters,nb_conv,strides=(1, 1),padding='valid',input_shape=(1,img_rows,img_cols)))
convout1=Activation('relu')
model.add(convout1)

model.add(Convolution2D(nb_filters,nb_conv))
convout2=Activation('relu')
model.add(convout2)

model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])


print("fiting the model")


model.fit(x_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test,y_test))

score=model.evaluate(x_test,y_test,verbose=0)

#predicting accuracy of code
print('test score:',score[0])
print('testt accuracy',score[1])






print(model.predict_classes(x_test[1:5]))
print(y_test[1:5])

#pickling a model for further use
model.save('peach_model.h5')

#testing for one immage
img_temp=Image.open('D:\\ty sem 6\\wce_hackathon\\New folder\\final\\peach\\Peach___Bacterial_spot1.jpg')
img_resize=img_temp.resize((img_rows,img_cols))
gray_img_temp=img_temp.convert('L')

result=model.predict_classes(gray_img_temp)

print(result)









