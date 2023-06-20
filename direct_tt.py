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
from keras import regularizers
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from pandas.core.frame import DataFrame
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def direct_tt(cv_folds, seed, Y1, X, snp_n, tt,channel,kernel,stride,unit,dropout, direct_epoch, direct_batch, direct_lr, o):
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)  ##参数--cv_folds --seed

    res = DataFrame()

    Y = Y1[tt].values
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
    best = float('inf')
    for training, vali in kfold.split(X, Y):
        X_train = X[training,]
        Y_train = Y[training]
        X_vali = X[vali,]
        Y_vali = Y[vali]
        for epoch in direct_epoch:  
            for batch in direct_batch:  
                for lr in direct_lr:  
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
                    y_test_pre = model.predict(X_test)
                    y_test2 = []
                    y_test_pre2 = []
                    for i in range(0, len(Y_test)):
                        y_test_pre2.append(float(y_test_pre[i]))
                        y_test2.append(float(Y_test[i]))
                    cor_test = pearsonr(y_test_pre2, y_test2)[0]
                    vali_cur = model.evaluate(X_vali, Y_vali)
                    file_path = "./" + o
                    try:
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)
                    except Exception as e:
                        print(e)

                    if best > vali_cur:
                        best = vali_cur
                        model.save(file_path + '/direct_' + tt + '.h5', overwrite=True)  ##参数--o
                    cor = ['Direct-prediction', 'CNN-'+tt, epoch, batch, lr,cor_test]
                    cor = DataFrame(cor)
                    cor =cor.transpose()
                    cor.columns=['Task','Model','Epoch','Batch_size','Learning_rate','Predictive_ability']
                    res = pd.concat([res, cor], sort=False)
                    print(cor)
                    outputpath = file_path + '/direct_res.csv'  ##参数--o
                    res.to_csv(outputpath, encoding="utf_8_sig", index=False)
                    K.clear_session()
                    gc.collect()
                    tf.keras.backend.clear_session()
