import numpy as np
np.random.seed(1337) #for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


##回归
# #create some data
# X = np.linspace(-1,1,200)
# np.random.shuffle(X)  #randomize the data
# Y = 0.5*X +2 + np.random.uniform(0,0.05,(200,))
# #plot data
# plt.scatter(X,Y)
# plt.xlim((-1.5,1.5))
# plt.ylim((1.2,2.6))
# plt.xticks(np.arange(-1.5,1.5,0.5))
# plt.yticks(np.arange(1.2,2.6,0.2))
# plt.show()
#
# X_train,Y_train = X[:160],Y[:160]
# X_test,Y_test = X[160:],Y[160:]
#
#
# #build a neural network from the 1st layer to the last layer
# model = Sequential()
# model.add(Dense(output_dim=1,input_dim=1))
#
# #choose loss function and optimizing method
# model.compile(loss='mse',optimizer='sgd')
#
# #training
# print('training -------------------------')
# for step in range(301):
#     cost = model.train_on_batch(X_train,Y_train)
#     if step%100==0:
#         print('train cost:',cost)
#
#
# #test
# print('testing --------------------------')
# test_cost = model.evaluate(X_test,Y_test,batch_size=40)
# print('test cost:',test_cost)
# W,b = model.layers[0].get_weights()
# print('Weights=',W,'\nbiases=',b)
#
# #plotting the prediction
# Y_pred = model.predict(X_test)
# plt.scatter(X_test,Y_test)
# plt.plot(X_test,Y_pred)
# plt.show()


# #classifier
# import numpy as np
# np.random.seed(1337) #for reproducibility
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense,Activation
# from keras.optimizers import RMSprop
#
# #download the mnist to the path'/.keras/datasets/'
# #X shape(60,000 28*28),y shape(10,000,)
# (X_train,y_train),(X_test,y_test) = mnist.load_data()
#
# #data pre-processing
# X_train = X_train.reshape(X_train.shape[0],-1)/255   #normalization
# X_test = X_test.reshape(X_test.shape[0],-1)/255
# y_train = np_utils.to_categorical(y_train,num_classes = 10)
# y_test = np_utils.to_categorical(y_test,num_classes = 10)
#
# #another way to bulid your neural net
# model = Sequential([
#     Dense(32,input_dim = 784),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax')
# ])
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
# print('training -------------------')
#
# #another way to train the model
# model.fit(X_train,y_train,nb_epoch = 2,batch_size = 32)
#
# print('\ntesting ------------------')
# #evaluate the model with the metrics we define earlier
# loss,accuracy = model.evaluate(X_test,y_test)
#
# print('test loss = ',loss)
# print('test accuracy = ',accuracy)



#CNN mnist
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=11, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)


