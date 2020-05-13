#!/usr/bin/env python3
import keras
import numpy as np
from numpy import asarray
from keras.models import Sequential, load_model
from keras.datasets import imdb
from keras.layers import Dense, Flatten, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import pandas as pd
import random
from scipy import stats
from sklearn.metrics import classification_report
from data_sort import *
from sklearn.model_selection import train_test_split
import time

def shuffle_dataframe(df,chunksize):
    [df_rows,df_columns] = df.shape
    numchunks = int(df_rows/chunksize)
    chunkidxs = random.sample(range(0,df_rows,chunksize),numchunks)
    shuffled_df = pd.DataFrame()
    for idx in chunkidxs:
        shuffled_df = shuffled_df.append(df.loc[idx:idx+chunksize-1])
    return shuffled_df

def create_segments_and_labels(df, time_steps, step, label_name):
    fingers = {'po': ['r', 't', 'y', 'u', 'f', 'g', 'h', 'j', 'v', 'b', 'n','m'], 'm': ['e', 'd', 'c','i','k'], 'r': ['w','s', 'x', 'o','.', 'l'], 'pi':['q','a','z','p'], 'space':['']}

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x'].values[i: i + time_steps]
        ys = df['y'].values[i: i + time_steps]
        zs = df['z'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        # label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        key = df['letter'].values[i]
        # print(i, df['letter'].values[i: i + time_steps])
        if pd.isna(key):
            labels.append(5)
        elif key =='none':
            labels.append(0)
        elif key[0].lower() in fingers.get('po'):
            labels.append(1)
        elif key[0].lower() in fingers.get('m'):
            labels.append(2)
        elif key[0].lower() in fingers.get('r'):
            labels.append(3)
        elif key[0].lower() in fingers.get('pi'):
            labels.append(4)
        else:
            print(key)
        segments.append([xs, ys, zs])

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

start = time.time()

np.random.seed(7)

f1 = 'data/alice_11_12'
f2 = 'data/alice_11_12_logkeys'
f3 = 'data/alphabet_02_19'
f4 = 'data/alphabet_02_19_logkeys'
f5 = 'data/alphabet_04_10'
f6 = 'data/alphabet_04_10_logkeys'
f7 = 'data/alphabet_04_12'
f8 = 'data/alphabet_04_12_logkeys'
f9 = 'data/alphabet_11_12'
f10 = 'data/alphabet_11_12_logkeys'
f11 = 'data/alphabet_4_10_2'
f12 = 'data/alphabet_4_10_2_logkeys'
f13 = 'data/ethics_04_24'
f14 = 'data/ethics_04_24_logkeys'
f15 = 'data/ethics_04_24_2'
f16 = 'data/ethics_04_24_2_logkeys'
f17 = 'data/gatsby_11_12'
f18 = 'data/gatsby_11_12_logkeys'
f19 = 'data/gibberish_11_12'
f20 = 'data/gibberish_11_12_logkeys'
f21 = 'data/iliad_11_12'
f22 = 'data/iliad_11_12_logkeys'
f23 = 'data/spaces_04_24'
f24 = 'data/spaces_04_24_logkeys'

# acc_files = [f5,f11,f23]
# lk_files = [f6,f12,f24]

model = None
model = load_model('theone.h5')

acc_files = [f1,f3,f5,f7,f9,f11,f17,f19,f21,f23]
lk_files = [f2,f4,f6,f8,f10,f12,f18,f20,f22,f24]

# accData entries look like [String time, String x, String y, String z]
# acc_entry_list = make_AccEntry_List(acc_files)

# lkData entries look like [String time, '>', String key]
# lk_entry_list = make_LKEntry_List(lk_files)

# letters to look for
checkLs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', '','.','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# how to classify the data
classes = ['none', 'po', 'm', 'r', 'pi','space']

# windows = make_dataframe(checkLs, acc_entry_list, lk_entry_list)
# windows = pd.read_csv('everything20.csv')
windows = pd.read_csv('ethics20.csv')
# print(windows['letter'].values[0:350])

# add_non_keypress(windows,acc_files,split=True)

[df_rows,df_columns] = windows.shape
print(windows.shape)

shuffled_windows = shuffle_dataframe(windows,chunksize=default_range)

if model == None:
    n_test = df_rows*0.33
    n_test = int(n_test - (n_test % default_range))

    n_train = df_rows*0.67
    n_train = int(n_train - (n_train % default_range))


    windows_test = shuffled_windows.tail(n=n_test)
    windows_train = shuffled_windows.head(n=n_train)

    pd.options.mode.chained_assignment = None  # default='warn'
    windows_train['x'] = windows_train['x'] / windows_train['x'].max()
    windows_train['y'] = windows_train['y'] / windows_train['y'].max()
    windows_train['z'] = windows_train['z'] / windows_train['z'].max()
    # Round numbers
    windows_train = windows_train.round({'x': 4, 'y': 4, 'z': 4})

    windows_test['x'] = windows_test['x'] / windows_test['x'].max()
    windows_test['y'] = windows_test['y'] / windows_test['y'].max()
    windows_test['z'] = windows_test['z'] / windows_test['z'].max()
    # Round numbers
    windows_test = windows_test.round({'x': 4, 'y': 4, 'z': 4})

    TIME_PERIODS=default_range
    STEP_DISTANCE=default_range
    LABEL='letter'

    x_train, y_train = create_segments_and_labels(windows_train,
                                                  TIME_PERIODS,
                                                  STEP_DISTANCE,
                                                  LABEL)

    x_test, y_test = create_segments_and_labels(windows_test,
                                                  TIME_PERIODS,
                                                  STEP_DISTANCE,
                                                  LABEL)

    print(n_test,n_train)

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
    # y_test_hot = to_categorical(y_test,num_classes)

    print('New y_train shape: ', y_train_hot.shape)
else:
    windows_test = shuffled_windows

    pd.options.mode.chained_assignment = None  # default='warn'
    windows_test['x'] = windows_test['x'] / windows_test['x'].max()
    windows_test['y'] = windows_test['y'] / windows_test['y'].max()
    windows_test['z'] = windows_test['z'] / windows_test['z'].max()
    # Round numbers
    windows_test = windows_test.round({'x': 4, 'y': 4, 'z': 4})

    TIME_PERIODS=default_range
    STEP_DISTANCE=default_range
    LABEL='letter'

    x_test, y_test = create_segments_and_labels(windows_test,
                                                  TIME_PERIODS,
                                                  STEP_DISTANCE,
                                                  LABEL)

    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    # set input and output dimensions
    num_time_periods, num_sensors = x_test.shape[1], x_test.shape[2]
    num_classes = len(classes)

    input_shape = (num_time_periods*num_sensors)
    x_test = x_test.reshape(x_test.shape[0],input_shape)

    print('x_test shape:', x_test.shape)
    print('input_shape:', input_shape)

    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')

    y_test_hot = to_categorical(y_test,num_classes)

    print('New y_test shape: ', y_test_hot.shape)

# create model
if model is None:
    embedding_vecor_length = 3
    model = Sequential()

    model.add(Reshape((default_range,3), input_shape=(input_shape,)))
    model.add(Dense(300,activation='selu'))
    model.add(Flatten())
    model.add(Dense(num_classes,activation='softmax'))
    print(model.summary())

    #model.add(Conv1D(filters=1, kernel_size=default_range, activation ='relu', input_shape=(x_train.shape[1],3)))

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    ]

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    BATCH_SIZE = default_range
    EPOCHS = 50
    history = model.fit(x_train, 
            y_train_hot, 
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            validation_split=.22,
            verbose=1)
            # validation_data=(x_test,y_test_hot))

# plt.figure(figsize=(6, 4))
# plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
# plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
# plt.plot(history.history['loss'], 'r--', label='Loss of training data')
# plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
# plt.title('Model Accuracy and Loss')
# plt.ylabel('Accuracy and Loss')
# plt.xlabel('Training Epoch')
# plt.ylim(0)
# plt.legend()
# plt.show()

# print(y_test[20:40])
y_pred_train = model.predict(x_test)
print(y_pred_train.shape)
# print(y_pred_train[20:40])
# for i in range(20,40):
    # print("actual: " + str(y_test[i]) + ", predicted: " + str(y_pred_train[i]))
max_y_pred_train = np.argmax(y_pred_train, axis=1)
# print(classification_report(y_test, max_y_pred_train))

none = 0
wnone = 0
mnone = 0
po = 0
wpo = 0
m = 0
wm = 0
r = 0
wr = 0
pi = 0
wpi = 0
fn = 0 
mpo = 0
mm = 0
mr = 0
mpi = 0
space = 0
wspace = 0
mspace = 0


for k in range(y_test.size):
    wrong = False
    if(max(y_pred_train[k]) == y_pred_train[k][0]):
        if(y_test[k] == 0):
            none += 1
        else:
            wnone += 1
            wrong = True
    elif(max(y_pred_train[k]) == y_pred_train[k][1]):
        if(y_test[k] == 1):
            po += 1
        else:
            wpo += 1
            wrong = True
    elif(max(y_pred_train[k]) == y_pred_train[k][2]):
        if(y_test[k] == 2):
            m += 1
        else: 
            wm += 1
            wrong = True
    elif(max(y_pred_train[k]) == y_pred_train[k][3]):
        if(y_test[k] == 3):
            r += 1
        else: 
            wr += 1
            wrong = True
    elif(max(y_pred_train[k]) == y_pred_train[k][4]):
        if(y_test[k] == 4):
            pi += 1
        else: 
            wpi += 1
            wrong = True
    elif(max(y_pred_train[k]) == y_pred_train[k][5]):
        if(y_test[k] == 5):
            space += 1
        else: 
            wspace += 1
            wrong = True
    if wrong:
        if(y_test[k] == 0):
            mnone += 1
        elif(y_test[k] == 1):
            mpo += 1
        elif(y_test[k] == 2):
            mm += 1
        elif(y_test[k] == 3):
            mr += 1
        elif(y_test[k] == 4):
            mpi += 1
        elif(y_test[k] == 4):
            mspace += 1
    
print("no finger correct: ", none)
print("no finger false: ", wnone)
print("no finger missed: ", mnone)
print("pointer correct: ", po)
print("pointer false: ", wpo)
print("pointer missed: ", mpo)
print("middle correct: ", m)
print("middle false: ",wm)
print("middle missed: ", mm)
print("ring correct: ", r)
print("ring false: ", wr)
print("ring missed: ", mr)
print("pinky correct: ", pi)
print("pinky false: ", wpi)
print("pinky missed: ", mpi)
print("space: ", space)
print("space false: ",wspace)
print("space missed: ", mspace)

print(none + mnone + wnone + po + wpo + mpo + m + wm + mm + r + wr + mr + pi + wpi + mpi + space + wspace + mspace)
print(none + wnone + po + wpo + m + wm + r + wr + pi + wpi + space + wspace)
end = time.time()
print(end - start)
    
'''
c1 = 0
c0 = 0
fp = 0
fn = 0

for k in range(y_test.size):
    if(y_test[k] == 1):
        if(y_pred_train[k][1] > .5):
            c1 += 1
        else:
            fn += 1
    else:
        if(y_pred_train[k][1] > .5):
            fp += 1
        else:
            c0 += 1
print("c1 = ", c1)
print("c0 = ", c0)
print("fp = ", fp)
print("fn = ", fn)

'''

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

#THIS CODE IS COPIED FIX LATER

