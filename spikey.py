#!/usr/bin/env python3
import keras
import numpy
from numpy import asarray
from keras.models import Sequential
from keras.datasets import imdb
from keras.layers import Dense, Flatten, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from data_sort import *

numpy.random.seed(7)

f1 = 'data/alphabet_04_10'
f2 = 'data/alphabet_04_10_logkeys'
f3 = 'data/alphabet_02_19'
f4 = 'data/alphabet_02_19_logkeys'

# (x_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
# print(y_train.size)
# print(x_train.size)

# accData entries look like [String time, String x, String y, String z]
acc_entry_list = make_AccEntry_List(f1) + make_AccEntry_List(f3)

# lkData entries look like [String time, '>', String key]
lk_entry_list = make_LKEntry_List(f2) + make_LKEntry_List(f4)

checkLs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
classes = ['key_press','non_key_press']
# window_list = []
# for letter in checkLs:
    # key_presses = list(filter(lambda x: x.key == letter, lk_entry_list))
    # for k in key_presses:
        # cur_window = Window(letter, acc_entry_list, get_index_of_matching_time(acc_entry_list, k.time)).window
        # cur_entry = []
        # for acc_entry in cur_window:
            # cur_entry.append(acc_entry.get_acceleration())
        # window_list.append(cur_entry)

#print(window_list)
windows = make_window_dict(checkLs, acc_entry_list, lk_entry_list)
add_non_keypress(windows,f1,split=True)

time_output = []
window_list = []
for key in windows:
    for window in windows[key]:
        i = 0
        while (len(time_output) != 0) and i < len(time_output) and window.window[0].get_time() > time_output[i][0]:
            i += 1
        time_output.insert(i, (window.window[0].get_time(), 0 if key == 'none' else 1))
        #window_list.insert(i,[])
        cur_entry = []
        for acc in window.window:
            #window_list[i].append(acc.get_acceleration())
            cur_entry.append(acc.get_acceleration())
        window_list.insert(i,cur_entry)

output = [item[1] for item in time_output]
#window_list_r = [window.window for window in window_list]
x_train = asarray(window_list[:int(len(window_list)/2)]).astype(np.float32)
x_test = asarray(window_list[int(len(window_list)/2):]).astype(np.float32)
y_train = asarray(output[:int(len(output)/2)])
y_test = asarray(output[int(len(output)/2):])

#x_train = to_categorical(x_train)
#x_test = to_categorical(window_array_test, 10)
#y_train = to_categorical(y_train,10)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# set input and output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = len(classes)

input_shape = (num_time_periods*num_sensors)
x_train = x_train.reshape(x_train.shape[0],input_shape)
x_test = x_test.reshape(x_test.shape[0],input_shape)

print('x_train shape:', x_train.shape)
print('input_shape:', input_shape)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

y_train_hot = to_categorical(y_train,num_classes)
y_test_hot = to_categorical(y_test,num_classes)

print('New y_train shape: ', y_train_hot.shape)

# create model
embedding_vecor_length = 3
model = Sequential()

model.add(Reshape((20,3), input_shape=(input_shape,)))
model.add(Dense(300,activation='selu'))
model.add(Flatten())
model.add(Dense(num_classes,activation='softmax'))
print(model.summary())

# model.add(Conv1D(filters=1, kernel_size=20, activation ='relu', input_shape=(x_train.shape[1],3)))

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

# model.add(Embedding(93, embedding_vecor_length, input_length=20))
# model.add(Embedding(93, embedding_vecor_length, input_length=20))
# model.add(Dropout(0.2))
#model.add(LSTM(100))
#model.add(Dropout(0.2))
# model.add(MaxPooling1D(pool_size=1))
# model.add(Flatten())
# model.add(Dense(100,activation='relu'))
# model.add(Dense(y_train.size, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
BATCH_SIZE = 20
EPOCHS = 50
history = model.fit(x_train, 
        y_train_hot, 
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        validation_split=0.2,
        verbose=1)
        # validation_data=(x_test,y_test_hot))



#plt.figure(figsize=(6, 4))
#plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
#plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
#plt.plot(history.history['loss'], 'r--', label='Loss of training data')
#plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
#plt.title('Model Accuracy and Loss')
#plt.ylabel('Accuracy and Loss')
#plt.xlabel('Training Epoch')
#plt.ylim(0)
#plt.legend()
#plt.show()

print(y_test[20:40])
y_pred_train = model.predict(x_test)
print(y_pred_train.shape)
print(y_pred_train[20:40])
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(y_test, max_y_pred_train))

# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
#FIT
#model.fit(window_array, epochs=1, batch_size=187, verbose = 0)
# evaluate model
#_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
#return accuracy



# apply filter to input data
#yhat = model.predict(window_array)
#print(model.get_weights())
