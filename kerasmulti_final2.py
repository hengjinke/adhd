import numpy as np
import json
import glob

import numpy as np
from numpy import *;
from scipy import interp
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.layers import merge, Input, concatenate
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, \
    AveragePooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *
from keras import regularizers
from keras import losses
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, concatenate
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
from sklearn.pipeline import Pipeline
# from quiver_engine import server
from keras.layers import Dropout
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras import callbacks
import time
from hyperopt.plotting import main_plot_vars
from hyperopt.plotting import main_plot_history
from hyperopt.plotting import main_plot_histogram
from hyperopt import base
import pandas
from keras import backend as K
import seaborn as sns
import pandas
import keras

sns.despine()
import json
from keras import losses

from sklearn.metrics import roc_auc_score
import sys
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.gpu_options.allow_growth = True


gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    
    tf.config.experimental.set_memory_growth(gpu, True)

warnings.filterwarnings('ignore')
#Train on 6850 samples, validate on 1713 samples
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
		
        font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 10,}

        plt.tick_params(labelsize=12)


        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', linestyle='-.', label='loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val_acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k',linestyle='--', label='val_loss')
        plt.grid(True)
        #plt.xlabel('Epoch', font1)
        #plt.ylabel('Acc-Loss', font1)
        font2 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 12,}
        #plt.title('Learning Curve', font2)
        plt.legend(loc="center right", prop=font1)
        plt.show()

dict = {
    'rows':'rows.csv',
	'columns':'columns.csv',
    'zs':'zs.csv'
	}

rowdf = pandas.read_csv(dict['rows'], delimiter=" ",header=None)
columndf = pandas.read_csv(dict['columns'], delimiter=" ",header=None)
zdf = pandas.read_csv(dict['zs'], delimiter=" ",header=None)
early_stopping = EarlyStopping(monitor='acc', patience=40)

def getaugmentumdata():
    X_train = pandas.read_csv('newData.csv', delimiter=" ",header=None);# unzip the newData.rar file downloaded from baidu cloud
    Y_train = pandas.read_csv('newLabel.csv', delimiter=" ",header=None);
    X_test = pandas.read_csv('X_test.csv', delimiter=" ",header=None);
    Y_test = pandas.read_csv('Y_test.csv', delimiter=" ",header=None);
    X_train = X_train.values;
    Y_train = Y_train.values;
    X_test = X_test.values;
    Y_test = Y_test.values;
    r = X_train[:, 0:64*20]
    c = X_train[:, 64*20:(64+30)*20]
    z = X_train[:, (64+30)*20:(64+30+64)*20]
    X_train = [r, c, z]

    rr = X_test[:, 0:64*20]
    cc = X_test[:, 64*20:(64+30)*20]
    zz = X_test[:, (64+30)*20:(64+30+64)*20]
    X_test = [rr, cc, zz]
    return X_train, X_test, Y_train, Y_test




callhistory = LossHistory()
early_stopping = EarlyStopping(monitor='acc', patience=10)
num_classes = 3
loss=0
accuracy = 0

X_train, X_test, Y_train, Y_test = getaugmentumdata();
def getMaxIndex(list1):
    ind = 0;
    max = -1
    for i in range(len(list1)):
        if list1[i] > max:
            max = list1[i];
    for i in range(len(list1)):
        if list1[i] == max:
            ind = i
    return ind
    
def getLabels(doubleArrays):
    for i in range(len(doubleArrays)):
        index = getMaxIndex(doubleArrays[i]);
        for j in range(len(doubleArrays[i])):
            if j == index:
                doubleArrays[i][j] = 1
            else:
                doubleArrays[i][j] = 0
    return doubleArrays;

def experiment(params):
    
    print ('Trying', params)
    f = open("kerasmulti_final_multi.txt", "a+")
    newcontent = json.dumps(params)
    
    try:

        row_input = Input((1280,), dtype='float32', name='row_input');
        row = Reshape((64, 20, 1), input_shape=(1280,))(row_input)
        row = Convolution2D(2, (2, 2))(row);
        row = Activation('relu')(row);
        row = Convolution2D(1, (2, 2))(row);
        row = Activation('tanh')(row);
        row = Convolution2D(2, (1, 1))(row);
        row = Activation('tanh')(row);
        row = Flatten()(row);
        row = Dense(500)(row)
        print(row.shape)

        column_input = Input((30*20,), dtype='float32', name='column_input');
        column = Reshape((30, 20, 1), input_shape=(30*20,))(column_input)
        column = Convolution2D(3, (2, 2))(column);
        column = Activation('sigmoid')(column);
        column = Convolution2D(1, (2, 2))(column);
        column = Activation('softplus')(column);
        column = Convolution2D(2, (1, 1))(column);
        column = Activation('relu')(column);
        print(column.shape)
        column = Flatten()(column);
        column = Dense(300)(column)

        z_input = Input((1280,), dtype='float32', name='z_input');
        z = Reshape((64, 20, 1), input_shape=(1280,))(z_input)
        z = Convolution2D(2, 2, 2)(z);
        z = Activation('sigmoid')(z);
        z = Convolution2D(2, 2, 2)(z);
        z = Activation('sigmoid')(z);
        z = Convolution2D(2, 1, 1)(z);
        z = Activation('sigmoid')(z);
        print(z.shape)
        z = Flatten()(z);
        z = Dense(500)(z)

        row_column = concatenate([row, column])
        print(row_column.shape)
        row_column_z = concatenate([row_column, z])
        print(row_column_z.shape)
        # x = Dense(500, activation='sigmoid')(row_column_z)
        x = Dropout(0.5)(row_column_z)
        x = Dense(260, activation='tanh')(x)
        x = Dense(60, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)


        model = Model(inputs=[row_input, column_input, z_input], outputs=[x]);
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss=losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        

        history = model.fit(X_train, Y_train,
                            epochs=40,
                            batch_size=500,
                            verbose=1, callbacks=[callhistory,early_stopping],
                            validation_data=(X_test, Y_test),
                            shuffle=True)
        newmmdpredict = model.predict(X_test)
        predictLabel = getLabels(newmmdpredict);
        loss,accuracy = model.evaluate(X_test,Y_test);
        from sklearn.metrics import classification_report
        report = classification_report(Y_test, predictLabel, output_dict=True)
        import collections;
        d = collections.OrderedDict(report["macro avg"])
        k = ["precision", "recall", "f1-score"]
        for i in range(3):
            newcontent += "    "+k[i]+":" + str(d[k[i]] * 100)
        callhistory.loss_plot('epoch')
    except Exception as e:
        print (str(e))
        print(repr(e))
        print ('-' * 10)

        return {'loss': 999999, 'status': STATUS_OK}
    newcontent += "    newmmdacc:" + str(accuracy * 100)
    newcontent += "\n"
    f.write(newcontent)
    f.close()
    print ('-' * 50)
    sys.stdout.flush()
    K.clear_session()
    
    return {'loss': loss, 'status': STATUS_OK}

space = {}
space = {'activation0':hp.choice('activation0',['tanh'])}


trials = Trials()
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
best = fmin(experiment, space, algo=tpe.suggest, max_evals=1, trials=trials)
domain = base.Domain(experiment, space)
#main_plot_vars(trials, bandit=domain)
#main_plot_history(trials, bandit=domain)
#main_plot_histogram(trials, bandit=domain)
print ('best: ')
print (best)




