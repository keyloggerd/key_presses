#!/usr/bin/env python3
import keras
import numpy
from numpy import asarray
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from data_sort import *
from keras.utils.np_utils import to_categorical
from keras.datasets import imdb

numpy.random.seed(7)

f1 = 'alphabet_02_19'
f2 = 'alphabet_02_19_logkeys'

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
print(y_train.size)
print(X_train.size)

# accData entries look like [String time, String x, String y, String z]
acc_entry_list = []
with open(f1) as f:  # accelerametor data
    content = f.read()
    accData = content.splitlines()[:-1]
    for i in accData:
        acc_entry_list.append(AccEntry(i))


# lkData entries look like [String time, '>', String key]
lk_entry_list = []
with open(f2) as f:  # logkey data
    content = f.read()
    pattern = re.compile("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \d+ - \d+ > \w")
    lkData = [a for a in content.splitlines() if pattern.match(a) is not None]
    for line in lkData:
        line = line.split()
        lk_entry_list.append(LKEntry(line[2]+"."+line[4], line[6]))


checkLs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
window_list = []
for letter in checkLs:
    key_presses = list(filter(lambda x: x.key == letter, lk_entry_list))
    for k in key_presses:
        cur_window = Window(letter, acc_entry_list, get_index_of_matching_time(acc_entry_list, k.time)).window
        cur_entry = []
        for acc_entry in cur_window:
            cur_entry.append(acc_entry.get_acceleration())
        window_list.append(cur_entry)

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
window_array_train = asarray(window_list[:int(len(window_list)/2)]).astype(np.float32)
w_test = asarray(window_list[int(len(window_list)/2):]).astype(np.float32)
y_train = asarray(output[:int(len(output)/2)])
y_test = asarray(output[int(len(output)/2):])

#window_array_train = to_categorical(window_array_train)
#w_test = to_categorical(window_array_test, 10)
#y_train = to_categorical(y_train,10)

#print(w_train.shape)
print(window_array_train.shape)
print(y_train.shape)

# create model
embedding_vecor_length = 3
model = Sequential()
model.add(Conv1D(filters=20, kernel_size=544, activation = 'relu', input_shape=(window_array_train.shape[1],3)))

model.add(Embedding(93, embedding_vecor_length, input_length=20))
model.add(Dropout(0.2))
#model.add(LSTM(100))
#model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(y_train.size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(window_array_train, y_train, epochs=3, batch_size=64)


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
