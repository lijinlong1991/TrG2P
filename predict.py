import argparse
import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import load_model
from pandas.core.frame import DataFrame

parser = argparse.ArgumentParser()

parser.add_argument("--geno", default=None, help="Training genotype *.txt file (m x n matrix)")
parser.add_argument("--model", default=None, help="Model *.h5 file")
parser.add_argument("-o", "--output", default=None, help="Override automatic results directory name")

args = parser.parse_args()

o = args.output
geno = args.geno
model = args.model


print("output:", o)
print("geno:", geno)
print("model:", model)


def predict(o,model,geno):
    print('************************* data loading ************************')
    X = pd.read_table(geno, header=None, delim_whitespace=True)
    model = load_model(model)
    print('************************* prediction ************************')
    pre = model.predict(X)
    pre2 = []
    for i in range(0, len(pre)):
        pre2.append(float(pre[i]))
    pre2 = pd.DataFrame(pre2)
    file_path = "./" + o
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
    except Exception as e:
        print(e)
    outputpath = file_path + '/prediction_res.csv'
    pre2.to_csv(outputpath, encoding="utf_8_sig", header=False,index=False)

predict(o,model,geno)
