import gc
import os

import numpy as np
import pandas as pd
from keras import backend as K
import keras
import random
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Conv1D, Input
from keras import regularizers
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from pandas.core.frame import DataFrame
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

def fine_tuning(cv_folds, seed, Y1, X, snp_n, tt, st,unit, dropout,ft_epoch, ft_batch, ft_lr, o):
    kfold5 = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)  ##参数--cv_folds --seed

    Y = Y1[tt].values  ##参数--tt
    Y[np.isnan(Y)] = np.nanmean(Y)
    Y = MinMaxScaler().fit_transform(Y.reshape(-1, 1))
    random.seed(seed)  ##参数 --seed
    tst_no = random.sample(range(0, len(Y)), round(len(Y) * 0.1))
    print('test_set:',tst_no)
    train_no = np.delete(range(0, len(Y)), tst_no, 0)
    X_test = X[tst_no,]
    Y_test = Y[tst_no]
    X = X[train_no,]
    Y = Y[train_no]
    res = DataFrame()

    for t in st:  ####参数 --st
        model_path = "./" + o
        model_raw = load_model(model_path + '/pretrain_' + t + '.h5')
        best = float('inf')
        for training, vali in kfold5.split(X, Y):
            X_train = X[training,]
            Y_train = Y[training]
            X_vali = X[vali,]
            Y_vali = Y[vali]
            for epoch in ft_epoch:  ##参数--ft_epoch
                for batch in ft_batch:  ##参数--ft_batch
                    for lr in ft_lr:  ##参数 --ft_lr
                        model_tr = Sequential()
                        for layer in model_raw.layers[:-3]:
                            model_tr.add(layer)
                            layer.trainable = False
                        model_tr.add(Dense(unit, kernel_initializer='normal', activation='relu'))
                        Dropout(dropout)
                        model_tr.add(Dense(unit, kernel_initializer='normal', activation='relu'))
                        Dropout(dropout)
                        model_tr.add(Dense(1, kernel_initializer='normal', name='dense_out'))
                        model_tr.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
                        model_tr.fit(X_train, Y_train, batch_size=batch, epochs=epoch, verbose=0)

                        file_path = "./" + o
                        try:
                            if not os.path.exists(file_path):
                                os.makedirs(file_path)
                        except Exception as e:
                            print(e)

                        vali_cur = model_tr.evaluate(X_vali, Y_vali)
                        if best > vali_cur:
                            best = vali_cur
                            model_tr.save(file_path + '/fine_tuned_model_' + t + '.h5', overwrite=True)  ##参数--o
                        y_test_pre = model_tr.predict(X_test)
                        y_test2 = []
                        y_test_pre2 = []
                        for i in range(0, len(Y_test)):
                            y_test_pre2.append(float(y_test_pre[i]))
                            y_test2.append(float(Y_test[i]))
                        cor_test = pearsonr(y_test_pre2, y_test2)[0]
                        cor = ['Fine_tuning', 'FT-'+t, epoch, batch, lr, cor_test]
                        cor = DataFrame(cor)
                        cor =cor.transpose()
                        cor.columns=['Task','Model','Epoch','Batch_size','Learning_rate','Predictive_ability']
                        res = pd.concat([res, cor], sort=False)
                        print(cor)
                        outputpath = file_path + '/fine_tuning_res.csv'  ##参数--o
                        res.to_csv(outputpath, encoding="utf_8_sig", index=False)
                        K.clear_session()
                        gc.collect()
