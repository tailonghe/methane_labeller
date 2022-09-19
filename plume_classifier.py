import tensorflow as tf
import keras
from tensorflow.keras.utils import Sequence
from keras.layers import Input, LSTM, Dense, Permute, Reshape, concatenate
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.metrics import mean_squared_error, BinaryIoU
from keras.models import Model, load_model, Sequential
from keras.losses import BinaryCrossentropy

import numpy as np
import os
import glob
from random import shuffle

# dR, rchannel, gchannel, bchannel, nirchannel, swir1channel, swir2channel, cloudscore
# NDVI, NDMI
SHAPE1 = SHAPE2 = 128
RES = 8
NUNIT = 1024
NPRED = 6

def cropping(_data, new_dim, axis=0):
    dim = _data.shape[axis]
    left = int((dim - new_dim)/2)
    right = int((dim + new_dim)/2)

    if axis == 0:
        return _data[left:right, :]
    elif axis == 1:
        return _data[:, left:right]

def padding(_data, new_dim, axis=0):
    '''
    Pad 2D field to accomadate the U-net.
    Zero-padding to the end of longitude and latitude dimensions to 
    '''
    res = new_dim - _data.shape[axis]
    left = np.floor(res/2).astype(int)
    right = np.ceil(res/2).astype(int)
    if axis == 0:
        return np.pad(_data, ((left, right),(0, 0)), 'constant')
    elif axis == 1:
        return np.pad(_data, ((0, 0),(left, right)), 'constant')
    
def resizeData(_data):
    tmp = _data.copy()

    while tmp.shape[0] != SHAPE1 or tmp.shape[1] != SHAPE2:
        if tmp.shape[0] > SHAPE1 or :
            tmp =  cropping(tmp, SHAPE1, 0)
        elif tmp.shape[0] < SHAPE1:
            tmp =  padding(tmp, SHAPE1, 0)
        elif tmp.shape[1] > SHAPE2:
            tmp =  cropping(tmp, SHAPE2, 1)
        elif tmp.shape[1] < SHAPE2:
            tmp =  padding(tmp, SHAPE2, 1)

    return tmp

def process_x(filename):
    _xdata = np.load(filename)['channels']
    tmp = np.zeros((SHAPE1, SHAPE2, _xdata.shape[-1]))
    for i in range(_xdata.shape[-1]):
        tmp[:, :, i] = resizeData(_xdata[:, :, i])
    
    out = np.zeros((SHAPE1, SHAPE2, NPRED))
    # dR, R, G, B, NDVI, NDMI
    out[:, :, :4] = tmp[:, :, :4]
    out[:, :, 4] = (tmp[:, :, 4] - tmp[:, :, 1])/(tmp[:, :, 4] + tmp[:, :, 1])
    out[:, :, 5] = (tmp[:, :, 5] - tmp[:, :, 6])/(tmp[:, :, 5] + tmp[:, :, 6])
    return out
    
def process_y(filename):
    _ydata = np.load(filename)['mask']
    tmp = np.zeros((SHAPE1, SHAPE2, 1))
    tmp[:, :, 0] = resizeData(_ydata)
    return tmp


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



class data_generator( Sequence ) :
    def __init__(self, fnames, batch_size=10):
        self.fnames = fnames
        self.batch_size = batch_size

    def __len__(self) :
        return np.ceil( len(self.fnames) / float(self.batch_size)).astype(int)

    def __getitem__(self, idx) :
        batch_f = self.fnames[ idx*self.batch_size : idx*self.batch_size + self.batch_size ]

        tempx = np.stack([ process_x(s) for s in batch_f])    # batch_size, 217, 175, 7
        tempy = np.stack([ process_y(s) for s in batch_f])

        return tempx, tempy


 # dimension is 217, 175

inputs = Input( ( SHAPE1, SHAPE2, NPRED ), name='model_input')

c1 = Conv2D(32, (3, 3), activation='tanh', padding='same', name='Block1_Conv1') (inputs)    # 120, 120
c1 = Conv2D(32, (3, 3), activation='tanh', padding='same', name='Block1_Conv2') (c1)   # 120, 120
p1 = MaxPooling2D((2, 2), name='Block1_MaxPool', padding='same') (c1)   # 60, 60

c2 = Conv2D(64, (3, 3), activation='tanh', padding='same', name='Block2_Conv1') (p1)   # 60, 60
c2 = Conv2D(64, (3, 3), activation='tanh', padding='same', name='Block2_Conv2') (c2)   # 60, 60
p2 = MaxPooling2D((2, 2), name='Block2_MaxPool', padding='same') (c2)   # 30, 30

c3 = Conv2D(128, (3, 3), activation='tanh', padding='same', name='Block3_Conv1') (p2)   # 30, 30
c3 = Conv2D(128, (3, 3), activation='tanh', padding='same', name='Block3_Conv2') (c3)   # 30, 30
p3 = MaxPooling2D((2, 2), name='Block3_MaxPool', padding='same') (c3)  # 15, 15

f5 = Reshape((int(SHAPE1*SHAPE2/RES**2), -1), name='Reshape') (p3)  # 900 x depth
lstm = LSTM(NUNIT, return_sequences=True, name='LSTM1') (f5)
resh = Reshape( (int(SHAPE1//RES), int(SHAPE2//RES), NUNIT) , name='Trans_Reshape') (lstm)

u4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='Block4_UpConv') (resh)  # 30, 30
u4_comb = concatenate([u4, c3])
c4 = Conv2D(128, (3, 3), activation='tanh', padding='same', name='Block4_Conv1') (u4_comb)  # 30, 30
c4 = Conv2D(128, (3, 3), activation='tanh', padding='same', name='Block4_Conv2') (c4)  # 30, 30

u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='Block5_UpConv') (c4)  # 60, 60
u5_comb = concatenate([u5, c2])
c5 = Conv2D(64, (3, 3), activation='tanh', padding='same', name='Block5_Conv1') (u5_comb)  # 60, 60
c5 = Conv2D(64, (3, 3), activation='tanh', padding='same', name='Block5_Conv2') (c5)  # 60, 60

u6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='Block6_UpConv') (c5)  # 120, 120
u6_comb = concatenate([u6, c1])
c6 = Conv2D(32, (3, 3), activation='tanh', padding='same', name='Block6_Conv1') (u6_comb)  # 120, 120
c6 = Conv2D(32, (3, 3), activation='tanh', padding='same', name='Block6_Conv2') (c6)  # 120, 120

outputs = Conv2D(1, (1, 1), activation='tanh', name='model_output') (c6)

# prepare model here
model = Model(inputs=[inputs], outputs=[outputs])

opt = Adam(lr=1e-5)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m, BinaryIoU])
model.summary()

# 800
trainfiles = np.genfromtxt('seed_12345_filelist.txt', dtype='str')
# train_DG = data_generator(trainfiles, batch_size=80)
X_train =  np.stack([ process_x(s) for s in trainfiles])    # batch_size, 217, 175, 7
Y_train = np.stack([ process_y(s) for s in trainfiles])


csv_logger = CSVLogger( 'plume_classification_log.csv', append=True, separator=';')
earlystopper = EarlyStopping(patience=100, verbose=1)
checkpointer = ModelCheckpoint('plume_classification_checkpt_{val_accuracy:.2f}.h5', verbose=1, save_best_only=True)

results = model.fit(X_train, 
                    Y_train, 
                    batch_size=120, 
                    epochs=250, 
                    validation_split=0.1,
                    shuffle=True, 
                    callbacks=[earlystopper, checkpointer, csv_logger])


model.save('plume_classification_model_seed_12345.h5')
