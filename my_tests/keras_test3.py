#coding:utf-8
#classifier
import numpy as np
np.random.seed(1337) #for reproducibility
from keras import regularizers,callbacks
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential,Model,load_model
from keras.layers import Input,Dense,Activation,Convolution2D,Flatten,MaxPooling2D,BatchNormalization,Dropout
from keras.optimizers import RMSprop,Adam

#download the mnist to the path'/.keras/datasets/'
#X shape(60,000 28*28),y shape(10,000,)
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# #data pre-processing
# print('type = ',type(X_train))
# print('shape = ',X_train.shape,X_train.shape[0],X_train.shape[1])
# X_train = X_train.reshape(X_train.shape[0],-1)/255   #normalization
# X_test = X_test.reshape(X_test.shape[0],-1)/255
# y_train = np_utils.to_categorical(y_train,num_classes = 10)
# y_test = np_utils.to_categorical(y_test,num_classes = 10)





# model = Sequential()
# model.add(Dense(32,input_dim = 28*28,activation = 'relu'))
# model.add(Dense(10,activation = 'softmax'))
#
#
# #another way to bulid your neural net
# model = Sequential([
#     Dense(32,input_dim = 784),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax')
# ])


# #不用序贯模型来构造FC层
# my_inputs = Input(shape = (28*28,))
# x = Dense(32,activation = 'relu')(my_inputs)
# x = Dense(32,activation = 'relu')(x)
# my_results = Dense(10,activation = 'softmax')(x)
# model = Model(inputs = my_inputs,outputs = my_results)
#
# #another way to define your optimizer
# rmsprop = RMSprop(lr = 0.001,rho = 0.9,epsilon = 1e-08,decay = 0.0)
#
# #we add metrics to get more results you want to see
# model.compile(
#     optimizer = rmsprop,
#     loss = 'categorical_crossentropy',
#     metrics = ['accuracy']
# )
#
#
#
#
#
# #可以使用 model.get_config()来查看构造好的模型参数,使用model.summary()来查看总体模型结构
# config = model.get_config()
# for x in config:
#     print(x)
# print('model.summary:------')
# model.summary()
#
#
#
# #another way to train the model
# model.fit(X_train,y_train,nb_epoch = 2,batch_size = 32)
#
# # #保存模型(可以使用'..\'表示上一级目录)
# # model.save('..\my_FCmodel.h5')
#
#
# print('\ntesting ------------------')
# #evaluate the model with the metrics we define earlier
# loss,accuracy = model.evaluate(X_test,y_test)
# print('parameters-----------------------------------------------')
# for i,x in enumerate(model.layers):
#     try:
#         print('type of cur_layer:',i,type(x.get_weights()))
#         print('shape of cur_layer:',i,np.array(x.get_weights()[0]).shape,np.array(x.get_weights()[1]).shape)
#         print(x.input_shape,x.output_shape)
#     except:
#         pass
#     print(x.get_weights(),'\n')
#
# print('test loss = ',loss)
# print('test accuracy = ',accuracy)
# import os
# cwd = os.getcwd()
# print(cwd)
# print(os.path.abspath(os.path.dirname(cwd)+'\.'))
#
# #加载保存好的模型
# my_model = load_model('..\my_FCmodel.h5')
# loss,accuracy = my_model.evaluate(X_test,y_test)
# print(loss,accuracy)




# #using CNN           test loss:  0.407238004947     test accuracy:  0.9768


X_train = X_train.reshape(-1,1,28,28)/255   #normalization
X_test = X_test.reshape(-1,1,28,28)/255
y_train = np_utils.to_categorical(y_train,num_classes = 10)
y_test = np_utils.to_categorical(y_test,num_classes = 10)

my_inputs = Input(shape = (1,28,28))
x = Convolution2D(
    batch_input_shape = (None,1,28,28),
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    data_format = 'channels_first',
)(my_inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = MaxPooling2D(
    pool_size = 2,
    strides = 2,
    padding = 'same',
    data_format = 'channels_first',
)(x)


x = Convolution2D(
    filters = 64,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    data_format = 'channels_first',
)(x)

x = BatchNormalization()(x)
x = Activation('relu')(x)


x = MaxPooling2D(
    pool_size = 2,
    strides = 2,
    padding = 'same',
    data_format = 'channels_first',
)(x)

x = Flatten()(x)
x = Dense(1024,activation = 'relu',kernel_regularizer = regularizers.l2(0.0001),activity_regularizer=regularizers.l1(0.0001))(x)
x = Dropout(0.4)(x)
my_outputs = Dense(10,activation = 'softmax')(x)

model = Model(inputs = my_inputs,outputs = my_outputs)



# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#可以使用 model.get_config()来查看构造好的模型参数,使用model.summary()来查看总体模型结构
config = model.get_config()
print(type(config),config)
for x in config:
    print(x)
print('model.summary:------')
model.summary()


print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=10, batch_size=64,callbacks = [callbacks.EarlyStopping(min_delta = 0.001,patience = 2,monitor = 'acc',mode = 'max')])

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

