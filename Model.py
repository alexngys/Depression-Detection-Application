import numpy as np
import keras 
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Activation, Dropout, MaxPooling2D, BatchNormalization, Conv1D, MaxPooling1D


path = 'dataset/'
features_train = np.load(path + 'train_samples_reg.npz', allow_pickle=True)['arr_0']
features_test = np.load(path + 'test_samples_reg.npz', allow_pickle=True)['arr_0']
labels_train = np.load(path + 'train_labels_reg.npz', allow_pickle=True)['arr_0']
labels_test = np.load(path + 'test_labels_reg.npz', allow_pickle=True)['arr_0']

X_train = np.array(features_train)
X_test = np.array(features_test)
Y_train = np.array(labels_train)
Y_test = np.array(labels_test)

# normalise
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
#X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])

def label_change(array):
    for i in range(len(array)):
        if array[i] < 6: # mild or no symptoms
            array[i] = 0
        elif array[i] >=6 and array[i] <=14: # moderate symptoms
            array[i] = 1
        elif array[i] >=15 and array[i] <=19: # moderately severe symptoms
            array[i] = 1
        else: # severe symptoms
            array[i] = 1
    return array

Y_train = label_change(Y_train)
Y_test = label_change(Y_test)


model = Sequential([
        Conv2D(32, kernel_size =(3, 5), strides =1, activation ='relu', input_shape = (32,235,1)), #(32,20,1)

        MaxPooling2D(pool_size =(3, 3), strides =(1, 3)),
        Conv2D(32, kernel_size =(9, 9), strides =1, activation ='relu'),

        MaxPooling2D(pool_size =(1, 3), strides =(1, 3)),
        Flatten(),
        Dense(512, activation ='relu'),

        Dropout(0.5),
        Dense(128, activation ='relu'),

        Dropout(0.5),
        Dense(2, activation ='softmax') 
    ])
'''
model = Sequential([
        Conv2D(32, kernel_size =(2,2), strides =1, activation ='relu', input_shape = (32,20,1)), #(32,20,1)

        MaxPooling2D(pool_size =(2, 2), strides =(1, 1)),
        Conv2D(32, kernel_size =(2, 2), strides =1, activation ='relu'),

        MaxPooling2D(pool_size =(2, 2), strides =(1, 1)),
        Flatten(),
        Dense(64, activation ='relu'),

        Dropout(0.5),
        Dense(32, activation ='relu'),

        Dropout(0.5),
        Dense(2, activation ='sigmoid') 
    ])
'''
# compile model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.summary()

# train model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=8, epochs=60) #56

figure, axis = plt.subplots(2)
line1, = axis[0].plot(history.history['loss'], 'r')
line2, = axis[0].plot(history.history['val_loss'], 'b')
axis[0].legend(['Train','Test'])
axis[0].set_title("Loss")

line3, = axis[1].plot(history.history['accuracy'], 'r')
line4, = axis[1].plot(history.history['val_accuracy'], 'b')
axis[1].legend(['Train','Test'])
axis[1].set_title("Accuracy")


plt.show()

model.save('my_model')