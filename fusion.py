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

def fusion(cv_folds, seed, Y1, X, snp_n, tt, st,unit,dropout, fu_epoch, fu_batch, fu_lr, o):
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)  ##参数--cv_folds --seed


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
    combine_model = list(st)  ##参数 --st
    #combine_model = list(combinations(model_list, 2))

    model_path = "./" + o

    model1 = load_model(model_path + '/fine_tuned_model_' + combine_model[0] + '.h5')
    model2 = load_model(model_path + '/fine_tuned_model_' + combine_model[1] + '.h5')
    model_tr1 = tr_model(model1)
    model_tr2 = tr_model(model2)
    best = float('inf')
    for training, vali in kfold.split(X, Y):
        X_train = X[training,]
        Y_train = Y[training]
        X_vali = X[vali,]
        Y_vali = Y[vali]
        for epoch in fu_epoch:  ##参数--fu_epoch
            for batch in fu_batch:  ##参数--fu_batch
                for lr in fu_lr:  ##参数 --fu_lr
                    model_tr = merge_model(snp_n, model_tr1, model_tr2,unit,dropout)
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
                        model_tr.save(file_path + '/fusion_model_' + combine_model[0] + '+' + combine_model[1] + '_.h5',
                                      overwrite=True)
                    y_test_pre = model_tr.predict(X_test)
                    y_test2 = []
                    y_test_pre2 = []
                    for i in range(0, len(Y_test)):
                        y_test_pre2.append(float(y_test_pre[i]))
                        y_test2.append(float(Y_test[i]))
                    cor_test = pearsonr(y_test_pre2, y_test2)[0]
                    cor = ['Fusion', 'FT-'+combine_model[0]+' + FT-'+combine_model[1], epoch, batch, lr, cor_test]
                    cor = DataFrame(cor)
                    cor =cor.transpose()
                    cor.columns=['Task','Model','Epoch','Batch_size','Learning_rate','Predictive_ability']
                    res = pd.concat([res, cor], sort=False)
                    print(cor)
                    outputpath = file_path + '/fusion_res.csv'
                    res.to_csv(outputpath, encoding="utf_8_sig", index=False)
                    K.clear_session()
                    gc.collect()

def tr_model(x):
    model_tr = Sequential()
    for layer in x.layers[:-2]: 
        model_tr.add(layer)
        layer.trainable = False
    return model_tr


def merge_model(snp_n_arg, model_tr1, model_tr2,unit,dropout):
    inp = Input((snp_n_arg,1))
    r1 = model_tr1(inp)
    r2 = model_tr2(inp)
    x = K.concatenate([r1,r2], axis=-1)
    model_tr=Dense(unit, kernel_initializer='normal', activation='relu')(x)
    model_tr=Dropout(dropout)(model_tr)
    model_tr=Dense(1, kernel_initializer='normal', name='dense_out')(model_tr)
    model =  keras.models.Model(inputs=inp, outputs=model_tr)
    return model