import gc
import os

import numpy as np
import pandas as pd
from keras import backend as K
import keras
import random
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Conv1D, Input, Flatten
from keras.optimizers import SGD
from keras import regularizers
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from pandas.core.frame import DataFrame
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

def pretraining(cv_folds, seed, Y1, X, snp_n, st,channel,kernel,stride,unit,dropout,pretrain_epoch, pretrain_batch, pretrain_lr, o):
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)  ##参数--cv_folds --seed

    res = DataFrame()

    for t in st:  ####参数 --st
        Y = Y1[t].values
        Y[np.isnan(Y)] = np.nanmean(Y)
        Y = MinMaxScaler().fit_transform(Y.reshape(-1, 1))
        best = float('inf')
        for training, vali in kfold.split(X, Y):
            X_train = X[training,]
            Y_train = Y[training]
            X_vali = X[vali,]
            Y_vali = Y[vali]
            for epoch in pretrain_epoch:  ##参数--pretrain_epoch
                for batch in pretrain_batch:  ##参数--pretrain_batch
                    for lr in pretrain_lr:  ##参数 --pretrain_lr
                        model = Sequential()
                        model.add(Conv1D(channel, kernel, strides=stride, input_shape=(snp_n, 1)))
                        model.add(Flatten())
                        model.add(Dense(unit, kernel_initializer='normal', activation='relu'))
                        Dropout(dropout)
                        model.add(Dense(unit, kernel_initializer='normal', activation='relu'))
                        Dropout(dropout)
                        model.add(Dense(1, kernel_initializer='normal'))
                        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
                        model.fit(X_train, Y_train, epochs=epoch, batch_size=batch, verbose=0)
                        y_vali_pre2 = []
                        y_vali2 = []
                        y_vali_pre = model.predict(X_vali)
                        for i in range(0, len(Y) - len(training)):
                            y_vali_pre2.append(float(y_vali_pre[i]))
                            y_vali2.append(float(Y_vali[i]))
                        cor_vali = pearsonr(y_vali_pre2, y_vali2)[0]
                        vali_cur = model.evaluate(X_vali, Y_vali)

                        file_path = "./" + o
                        try:
                            if not os.path.exists(file_path):
                                os.makedirs(file_path)
                        except Exception as e:
                            print(e)

                        if best > vali_cur:
                            best = vali_cur
                            model.save(file_path + '/pretrain_' + t + '.h5', overwrite=True)  ##参数--o
                        cor = ['Pretraining', 'CNN-'+t, epoch, batch, lr, cor_vali]
                        cor = DataFrame(cor)
                        cor =cor.transpose()
                        cor.columns=['Task','Model','Epoch','Batch_size','Learning_rate','Predictive_ability']
                        res = pd.concat([res, cor], sort=False)
                        print(cor)
                        outputpath = file_path + '/pretaining_res.csv'  ##参数--o
                        res.to_csv(outputpath, encoding="utf_8_sig", index=False)
                        K.clear_session()
                        gc.collect()
                        tf.keras.backend.clear_session()
