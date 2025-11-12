# Prototype Retrieval-Augmented Test-Time Optimization  for Federated Intrusion Detection System

## Acknowledgement

All baselines and our method are implemented on top of [PFLlib](https://www.pfllib.com/benchmark.html). 

We are very grateful for their outstanding library.


## Dataset

All used datasets can be downloaded [here](https://drive.google.com/file/d/1eV5628osYru5ndeqoEPv9mGIKaI29BHG/view?usp=sharing)

Please unzip the RAR file into the "dataset" folder.


## Quickly Start  (baseline FL algorithms + our proposed FedPRO)

You can modify the '-data' parameter to specify the dataset and β value, i.e., {dataset folder}_{β values}

#### NSL-KDD (Personalized test, β=0.5)
```
python system/main.py -data NSLKDD_0.5 -algo Local -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # Local
python system/main.py -data NSLKDD_0.5 -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # FedAvg
python system/main.py -data NSLKDD_0.5 -algo MOON -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # MOON
python system/main.py -data NSLKDD_0.5 -algo FedProto -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # FedProto
python system/main.py -data NSLKDD_0.5 -algo FedTGP -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # FedTGP
python system/main.py -data NSLKDD_0.5 -algo GPFL -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # GPFL
python system/main.py -data NSLKDD_0.5 -algo FedDBE -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # FedDBE
```
#### NSL-KDD  (Global test, β==0.5)
```
python system/main.py -data NSLKDD_global_0.5 -algo Local -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # Local
python system/main.py -data NSLKDD_global_0.5 -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # FedAvg
python system/main.py -data NSLKDD_global_0.5 -algo MOON -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # MOON
python system/main.py -data NSLKDD_global_0.5 -algo FedProto -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # FedProto
python system/main.py -data NSLKDD_global_0.5 -algo FedTGP -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # FedTGP
python system/main.py -data NSLKDD_global_0.5 -algo GPFL -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # GPFL
python system/main.py -data NSLKDD_global_0.5 -algo FedDBE -nc 5 -ncl 5 -topk 1 -fd 122 --skip_FL True # FedDBE
```

#### CICIDS-2018 (Personalized test, β=0.5)
```
python system/main.py -data CICIDS_0.5 -algo Local -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # Local
python system/main.py -data CICIDS_0.5 -algo FedAvg -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # FedAvg
python system/main.py -data CICIDS_0.5 -algo MOON -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # MOON
python system/main.py -data CICIDS_0.5 -algo FedProto -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # FedProto
python system/main.py -data CICIDS_0.5 -algo FedTGP -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # FedTGP
python system/main.py -data CICIDS_0.5 -algo GPFL -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # GPFL
python system/main.py -data CICIDS_0.5 -algo FedDBE -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # FedDBE
```

#### CICIDS-2018 (Global test, β=0.5)
```
python system/main.py -data CICIDS_global_0.5 -algo Local -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # Local
python system/main.py -data CICIDS_global_0.5 -algo FedAvg -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # FedAvg
python system/main.py -data CICIDS_global_0.5 -algo MOON -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # MOON
python system/main.py -data CICIDS_global_0.5 -algo FedProto -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # FedProto
python system/main.py -data CICIDS_global_0.5 -algo FedTGP -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # FedTGP
python system/main.py -data CICIDS_global_0.55 -algo GPFL -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # GPFL
python system/main.py -data CICIDS_global_0.5 -algo FedDBE -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True # FedDBE
```
