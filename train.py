import argparse
import pandas as pd
import numpy as np

from direct_tt import direct_tt
from pretrain import pretraining
from fine_tune import fine_tuning
from fusion import fusion
from summary import summary

# 创建参数解析器
parser = argparse.ArgumentParser()

# 添加命令行参数
parser.add_argument("-s", "--seed", type=int, default=42, help="Value of random and cv seeds")
parser.add_argument("-o", "--output", default=None, help="Override automatic results directory name")
parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds to use in cross-validation")
parser.add_argument("--geno", default="./data/geno.txt", help="Training genotype *.txt file (m x n matrix)")
parser.add_argument("--pheno", default="./data/pheno.txt", help="Training phenotype *.txt file (m x 1 vector)")
parser.add_argument("--st", nargs="+", default=['PH','GW'], help="Source domain traits, n=2")
parser.add_argument("--tt", default="YLD", help="Target domain trait, n=1")
parser.add_argument("--channel", type=int, default=1, help="The number of out channels in convolutional layer")
parser.add_argument("--kernel", type=int, default=10, help="The kernel size in convolutional layer")
parser.add_argument("--stride", type=int, default=10, help="The stride in convolutional layer")
parser.add_argument("--unit", type=int, default=64, help="The units in fully connected layers")
parser.add_argument("--dropout", type=float, default=0.3, help="The dropout in fully connected layers")
parser.add_argument("--pretrain_epoch", nargs="+", default=[40,], help="Pretraining epoch size")
parser.add_argument("--pretrain_batch", nargs="+", default=[40,], help="Pretraining batch size")
parser.add_argument("--pretrain_lr", nargs="+", default=[0.001,], help="Pretraining learning rate")
parser.add_argument("--ft_epoch", nargs="+", default=[40,], help="Fine-turning epoch size")
parser.add_argument("--ft_batch", nargs="+", default=[40,], help="Fine-turning batch size")
parser.add_argument("--ft_lr", nargs="+", default=[0.001,], help="Fine-turning learning rate")
parser.add_argument("--fu_epoch", nargs="+", default=[40,], help="Fusion epoch size")
parser.add_argument("--fu_batch", nargs="+", default=[40,], help="Fusion batch size")
parser.add_argument("--fu_lr", nargs="+", default=[0.001,], help="Fusion learning rate")
parser.add_argument("--direct_epoch", nargs="+", default=[40,], help="Direct prediction epoch size")
parser.add_argument("--direct_batch", nargs="+", default=[40,], help="Direct prediction batch size")
parser.add_argument("--direct_lr", nargs="+", default=[0.001,], help="Direct prediction learning rate")

# 解析命令行参数
args = parser.parse_args()

# 打印参数
seed = args.seed
o = args.output
cv_folds = args.cv_folds
geno = args.geno
pheno = args.pheno
st = args.st
tt = args.tt
channel =  args.channel
kernel =  args.kernel
stride =  args.stride
unit =  args.unit
dropout =  args.dropout
pretrain_epoch = [int(x) for x in args.pretrain_epoch]
pretrain_batch = [int(x) for x in args.pretrain_batch]
pretrain_lr = [float(x) for x in args.pretrain_lr]
ft_epoch = [int(x) for x in args.ft_epoch]
ft_batch = [int(x) for x in args.ft_batch]
ft_lr = [float(x) for x in args.ft_lr]
fu_epoch = [int(x) for x in args.fu_epoch]
fu_batch = [int(x) for x in args.fu_batch]
fu_lr = [float(x) for x in args.fu_lr]
direct_epoch = [int(x) for x in args.direct_epoch]
direct_batch = [int(x) for x in args.direct_batch]
direct_lr = [float(x) for x in args.direct_lr]

print("seed:", seed)
print("output:", o)
print("cv_folds:", cv_folds)
print("geno:", geno)
print("pheno:", pheno)
print("st:", st)
print("tt:", tt)
print("channel:", channel)
print("kernel:", kernel)
print("stride:", stride)
print("unit:", unit)
print("dropput:", dropout)
print("pretrain_epoch:", pretrain_epoch)
print("pretrain_batch:", pretrain_batch)
print("pretrain_lr:", pretrain_lr)
print("ft_epoch:", ft_epoch)
print("ft_batch:", ft_batch)
print("ft_lr:", ft_lr)
print("fu_epoch:", fu_epoch)
print("fu_batch:", fu_batch)
print("fu_lr:", fu_lr)
print("direct_epoch:", direct_epoch)
print("direct_batch:", direct_batch)
print("direct_lr:", direct_lr)

def main_function(cv_folds, seed, pheno, geno, tt, st,channel,kernel,stride,unit,dropout,direct_epoch, direct_batch, direct_lr,pretrain_epoch,
                  pretrain_batch, pretrain_lr,ft_epoch, ft_batch, ft_lr,fu_epoch, fu_batch, fu_lr, o):
    print('****************************** Data loading *****************************************')
    Y1 = pd.read_csv(pheno, header=0)  ##参数 --pheno
    X = pd.read_table(geno, header=None, delim_whitespace=True)  ##参数 --geno
    snp_n = len(X.columns)
    X = np.expand_dims(X, axis=2)
    X = X.astype(np.float32)

    print('************************ Target trait direct prediction via CNN ***********************')
    direct_tt(cv_folds, seed, Y1, X, snp_n, tt, channel, kernel, stride, unit, dropout, direct_epoch, direct_batch, direct_lr, o)
    print('************************ Source traits pretraining via CNN ****************************')
    pretraining(cv_folds, seed, Y1, X, snp_n, st, channel, kernel, stride, unit, dropout, pretrain_epoch, pretrain_batch, pretrain_lr, o)
    print('******************************* Fine_tuning  ******************************************')
    fine_tuning(cv_folds, seed, Y1, X, snp_n, tt, st,unit, dropout,ft_epoch, ft_batch, ft_lr, o)
    print('******************************* Fusion model  *****************************************')
    fusion(cv_folds, seed, Y1, X, snp_n, tt, st, unit, dropout, fu_epoch, fu_batch, fu_lr, o)
    print('*********************************  Summary   *******************************************')
    summary(o)

main_function(cv_folds, seed, pheno, geno, tt, st,channel, kernel, stride, unit, dropout, direct_epoch, direct_batch, direct_lr,pretrain_epoch,
                  pretrain_batch, pretrain_lr,ft_epoch, ft_batch, ft_lr,fu_epoch, fu_batch, fu_lr, o)