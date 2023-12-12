# TrG2P: A transfer learning based approach to improve genomic prediction

TrG2P is a novel strategy to improve genomic prediction (GP) performance based on transfer learning.  TrG2P aims to get knowledge from the source task's learning system to improve the target GP task.  Two main functions were designed in this approach, including "train.py" and "predict.py", respectively. Three processes were applied by "train.py" function, including pretraining, fine-tuning and building fusion model. The "predict.py" function  is used to predict genomic estimated breeding values (GEBVs) for new accessions.

----------------------------------------
## Environment
This package was builded with the followed packages:
```c
python 3.9
tensorflow 2.11.0
keras 2.11.0
scikit-learn 1.0.2
pandas 1.5.3
numpy  1.24.2
```
## Data format
### genotype
- The genotype file need to contain all markers in a "*.txt" file, including m accessions and n markers. 

|            | markers |   |   |     |
|:--:|:--:|:--:|:--:|:--:|
| **accessions** | 1      | 0 | 0 | -1  |
|            | 0      | 1 | 1 | 0   |
|            | -1     | 1 | 0 | -1  |

### phenotype
- The phenotype file need to contain mutiple traits in a "*.csv" file, and the header is the trait's name. 



|            | traits |       |      |       |
|:--:|:--:|:--:|:--:|:--:|
| **accessions** | GHID   | PH    | PnN  | LA    |
|            | A1257  | 118.8 | 12.9 | 21.1  |
|            | A1258  | 125   | 11.8 | 27.6  |
|            | A1302  | 117.7 | 12.5 | 26    |




## Parameter list
### train.py

| Parameter      | Type           | Default          | Description                                        |
|:--:|:--:|:--:|:--:|
| seed           | int            | 42               | Value of random and cv seeds                       |
| output              | directory      | None             | Override automatic results directory name          |
| cv_folds       | int            | 5                | Number of folds to use in cross-validation         |
| geno           | directory      | ./data/geno.txt  | Training genotype *.txt file (m x n matrix)        |
| pheno          | directory      | ./data/pheno.csv | Training phenotype *.csv file (m x i vector)       |
| st             | character, n=2 | None             | Source domain traits, n=2                          |
| tt             | character, n=1 | None             | Target domain trait, n=1                           |
| channel        | int            | 1                | The number of out channels in convolutional layer  |
| kernel         | int            | 10               | The kernel size in convolutional layer             |
| stride         | int            | 10               | The stride in convolutional layer                  |
| unit           | int            | 64               | The units in fully connected layers                |
| dropout        | float          | 0.3              | The dropout in fully connected layers              |
| pretrain_epoch | int list       | 40               | Pretraining epoch                             |
| pretrain_batch | int list       | 40               | Pretraining batch size                             |
| pretrain_lr    | float list     | 0.001            | Pretraining learning rate                          |
| ft_epoch       | int list       | 40               | Fine-turning epoch                            |
| ft_batch       | int list       | 40               | Fine-turning batch size                            |
| ft_lr          | float list     | 0.001            | Fine-turning learning rate                         |
| fu_epoch       | int list       | 40               | Fusion epoch                                  |
| fu_batch     | int list   | 40    | Fusion batch size                |
| fu_lr        | float list | 0.001 | Fusion learning rate             |
| direct_epoch | int list   | 40    | Direct prediction epoch     |
| direct_batch | int list   | 40    | Direct prediction batch size     |
| direct_lr    | float list | 0.001 | Direct prediction learning rate  |

### predict.py
| Parameter | Type      | Default         | Description                                  |
|:--:|:--:|:--:|:--:|
| output    | directory | None            | Override automatic results directory name    |
| geno      | directory | ./data/geno.txt | Training genotype *.txt file (m x n matrix)  |
| model     | directory | None            | Model *.h5 file                              |

## Demo
### Modeling
```c
python train.py -s 42 --output results --cv_folds 5 --geno ./data/rice299_geno2.txt --pheno ./data/rice299_phe.csv --st YPP GW --tt YLD --pretrain_epoch 60 --pretrain_batch 20  --pretrain_lr 0.001 --ft_epoch  60 --ft_batch 20 --ft_lr 0.001 --fu_epoch  60 --fu_batch 20  --fu_lr  0.001 --direct_epoch 40 --direct_batch 40 --direct_lr 0.001
## epoch, batch, lr can be a list
```
11 files will be outputed by train.py program, including:
- **direct_*.h5**,  Direct prediction model for the target trait based on CNN.
- **direct_res.csv**, Direct predictive ability for the target trait based on CNN.
- **pretrain_*.h5**, Pretraining model for the source traits based on CNN, 2 models will be outputed.
- **pretaining_res.csv**, Pretraining predictive ability for the source trait based on CNN.
-  **fine_tuned_model_*.h5**, Fine-tuned model for the target traits based on transfer learning, 2 models will be outputed.
- **fine_tuning_res.csv**, Fine-tuning predictive ability for the target trait based on transfer learning.
- **fusion_model_*+*.h5**, Fusion model for the target traits based on transfer learning.
- **fusion_res.csv**, Fusion model's predictive ability for the target trait based on transfer learning.
- **results_summary.csv**, The summary of all results for the target trait.

### Predicting with the trained model
```c
python predict.py  --output results  --geno "./data/rice299_geno2.txt"  --model ./results/fine_tuned_model_PH.h5
##  for "--geno", a new genotype file should be used, here just an example.
```
As a result, 1 files will be outputed by predict.py program, 
- **prediction_res.csv**, The predictive values with known genotype.
