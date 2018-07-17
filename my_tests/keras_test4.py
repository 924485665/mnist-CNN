#coding:utf-8

#实现inception v2的卷积神经网络
import numpy as np
np.random.seed(1337) #for reproducibility
from keras import regularizers,callbacks
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential,Model,load_model
from keras.layers import Input,Dense,Activation,Convolution2D,Flatten,MaxPooling2D,BatchNormalization,Dropout,concatenate
from keras.optimizers import RMSprop,Adam

#download the mnist to the path'/.keras/datasets/'
#X shape(60,000 28*28),y shape(10,000,)
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#pre-processing
X_train = X_train.reshape(-1,28,28,1)/255   #normalization
X_test = X_test.reshape(-1,28,28,1)/255
y_train = np_utils.to_categorical(y_train,num_classes = 10)
y_test = np_utils.to_categorical(y_test,num_classes = 10)

#creating model
my_inputs = Input(shape = (28,28,1))
x = Convolution2D(
    batch_input_shape = (None,28,28,1),
    filters = 64,
    kernel_size = 1,
    strides = 1,
    padding = 'same',
    data_format = 'channels_last',
)(my_inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

#第一个分支，1*1卷积核再接3*3卷积核
#1*1卷积核
branch1 = Convolution2D(
    batch_input_shape = (None,28,28,1),
    filters = 64,
    kernel_size = 1,
    strides = 1,
    padding = 'same',
    data_format = 'channels_last',
)(my_inputs)
branch1 = BatchNormalization()(branch1)
branch1 = Activation('relu')(branch1)

#3*3卷积核
branch1 = Convolution2D(
    filters = 64,
    kernel_size = 3,
    strides = 1,
    padding = 'same',
    data_format = 'channels_last',
)(branch1)
branch1 = BatchNormalization()(branch1)
branch1 = Activation('relu')(branch1)



#第二个分支，1*1卷积核再接5*5卷积核
#1*1卷积核
branch2 = Convolution2D(
    batch_input_shape = (None,28,28,1),
    filters = 64,
    kernel_size = 1,
    strides = 1,
    padding = 'same',
    data_format = 'channels_last',
)(my_inputs)
branch2 = BatchNormalization()(branch2)
branch2 = Activation('relu')(branch2)

#5*5卷积核
branch2 = Convolution2D(
    filters = 64,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    data_format = 'channels_last',
)(branch2)
branch2 = BatchNormalization()(branch2)
branch2 = Activation('relu')(branch2)

#第三个分支，3*3最大池化再接1*1卷积
branch3 = MaxPooling2D(
    batch_input_shape = (None,28,28,1),
    pool_size = 3,
    strides = 1,
    padding = 'same',
    data_format = 'channels_last'
)(my_inputs)

#再接1*1卷积
branch3 = Convolution2D(
    filters = 64,
    kernel_size = 1,
    strides = 1,
    padding = 'same',
    data_format = 'channels_last'
)(branch3)
branch3 = BatchNormalization()(branch3)
branch3 = Activation('relu')(branch3)


#将第一层的所有结果拼接起来
first_layer_output = concatenate([x,branch1,branch2,branch3],axis = -1)
#Flatten
x = Flatten()(first_layer_output)

x = Dense(1024,activation = 'relu',kernel_regularizer = regularizers.l2(0.0001),activity_regularizer=regularizers.l1(0.0001))(x)
x = Dropout(0.4)(x)

my_outputs = Dense(10,activation = 'softmax')(x)

inception2 = Model(inputs = my_inputs,outputs = my_outputs)


#compiling model
my_adam = Adam(lr = 1e-4)
inception2.compile(optimizer=my_adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
#training model
inception2.fit(X_train, y_train, epochs=3, batch_size=64,validation_split = 0.1,callbacks = [callbacks.EarlyStopping(monitor = 'acc',mode = 'max')])

#saving model(可以使用'..\'表示上一级目录)
#model.save('..\inception2.h5')

print('\nTesting ------------')
# testing model
loss, accuracy = inception2.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)