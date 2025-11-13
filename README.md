# Prototype Retrieval-Augmented Federated Learning System for Robust Intrusion Detection



## Acknowledgement

All baselines and our method are implemented on top of [PFLlib](https://www.pfllib.com/benchmark.html). 

We are very grateful for their outstanding library.


## Dataset

All used datasets can be downloaded [here](https://drive.google.com/file/d/1eV5628osYru5ndeqoEPv9mGIKaI29BHG/view?usp=sharing)

Please unzip the RAR file into the "system/dataset" folder.

## Quickly Start (baseline FL algorithms + our proposed FedPRO)

**In the following bash examples, we use β = 0.5 as a running example.**

**Please modify the `-data` argument to select a dataset with a different dataset and β value, using the format `{dataset folder}_{β value}`.**



#### NSL-KDD (Personalized test)
```
python system/main.py -data NSLKDD_0.5 -algo Local -ncl 5 -topk 1 --skip_FL True # Local
python system/main.py -data NSLKDD_0.5 -algo FedAvg -ncl 5 -topk 1 --skip_FL True # FedAvg
python system/main.py -data NSLKDD_0.5 -algo MOON -ncl 5 -topk 1 --skip_FL True # MOON
python system/main.py -data NSLKDD_0.5 -algo FedProto -ncl 5 -topk 1 --skip_FL True # FedProto
python system/main.py -data NSLKDD_0.5 -algo FedTGP -ncl 5 -topk 1 --skip_FL True # FedTGP
python system/main.py -data NSLKDD_0.5 -algo GPFL -ncl 5 -topk 1 --skip_FL True # GPFL
python system/main.py -data NSLKDD_0.5 -algo FedDBE -ncl 5 -topk 1 --skip_FL True # FedDBE
```
#### NSL-KDD  (Global test ）
```
python system/main.py -data NSLKDD_global_0.5 -algo Local -ncl 5 -topk 1 --skip_FL True # Local
python system/main.py -data NSLKDD_global_0.5 -algo FedAvg -ncl 5 -topk 1 --skip_FL True # FedAvg
python system/main.py -data NSLKDD_global_0.5 -algo MOON -ncl 5 -topk 1 --skip_FL True # MOON
python system/main.py -data NSLKDD_global_0.5 -algo FedProto -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # FedProto
python system/main.py -data NSLKDD_global_0.5 -algo FedTGP -ncl 5 -topk 1 --skip_FL True # FedTGP
python system/main.py -data NSLKDD_global_0.5 -algo GPFL -ncl 5 -topk 1 --skip_FL True # GPFL
python system/main.py -data NSLKDD_global_0.5 -algo FedDBE -ncl 5 -topk 1 --skip_FL True # FedDBE
```

#### CICIDS-2018 (Personalized test)
```
python system/main.py -data CICIDS_0.5 -algo Local  -ncl 7 -topk 1  --skip_FL True # Local
python system/main.py -data CICIDS_0.5 -algo FedAvg  -ncl 7 -topk 1 --skip_FL True # FedAvg
python system/main.py -data CICIDS_0.5 -algo MOON  -ncl 7 -topk 1 --skip_FL True # MOON
python system/main.py -data CICIDS_0.5 -algo FedProto  -ncl 7 -topk 1  --skip_FL True # FedProto
python system/main.py -data CICIDS_0.5 -algo FedTGP  -ncl 7 -topk 1  --skip_FL True # FedTGP
python system/main.py -data CICIDS_0.5 -algo GPFL  -ncl 7 -topk 1  --skip_FL True # GPFL
python system/main.py -data CICIDS_0.5 -algo FedDBE  -ncl 7 -topk 1  --skip_FL True # FedDBE
```

#### CICIDS-2018 (Global test)
```
python system/main.py -data CICIDS_global_0.5 -algo Local -ncl 7 -topk 1  --skip_FL True # Local
python system/main.py -data CICIDS_global_0.5 -algo FedAvg  -ncl 7 -topk 1  --skip_FL True # FedAvg
python system/main.py -data CICIDS_global_0.5 -algo MOON -nc -ncl 7 -topk 1  --skip_FL True # MOON
python system/main.py -data CICIDS_global_0.5 -algo FedProto  -ncl 7 -topk 1  --skip_FL True # FedProto
python system/main.py -data CICIDS_global_0.5 -algo FedTGP -ncl 7 -topk 1 --skip_FL True # FedTGP
python system/main.py -data CICIDS_global_0.55 -algo GPFL -ncl 7 -topk 1  --skip_FL True # GPFL
python system/main.py -data CICIDS_global_0.5 -algo FedDBE  -ncl 7 -topk 1  --skip_FL True # FedDBE
```



## Experiments on additional large-class datasets (BODMAS, Ominiglot)

#### BODMAS example

```
python main.py -data BODMAS_0.1 -m cicids -algo FedAvg -ncl 267 -skip_FL True -topk 1 # Personalized test
python main.py -data BODMAS_global_0.1 -m cicids -algo FedAvg -ncl 267 -skip_FL True -topk 1 # Global test
```

#### Omniglot example

```
python main.py -data Omniglot_0.1 -m ResNet10 -algo FedAvg -ncl 1623 -skip_FL True -topk 1 # Personalized test
python main.py -data Omniglot_global_0.1 -m ResNet10 -algo FedAvg -ncl 1623 -skip_FL True -topk 1 # Global test
```

