# Prototype Retrieval-Augmented Test-Time Optimization  for Federated Intrusion Detection System

All baselines and our method are implemented on top of [PFLlib](https://www.pfllib.com/benchmark.html). 
We are very grateful for this outstanding library.



## Dataset

All used datasets can be downloaded [here](https://drive.google.com/file/d/1mS2fbBCeXSvNeOlrvd0sOme2uUgTKqpJ/view?usp=sharing)

Please unzip the dataset.rar file into the dataset folder.


## Quickly Start (baseline FL algorithms + our proposed FedPRO)

```
# The example case uses FedAvg as the baseline algorithm.
# You can modify the "-algo" parameter to specify the other FL approaches.

cd system 
# NSL-KDD (Personalized test)
python main.py -data NSLKDD_0.1 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=0.1
python main.py -data NSLKDD_0.5 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=0.5
python main.py -data NSLKDD_1 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=1
python main.py -data NSLKDD_10 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=10

# NSL-KDD  (Global test)
python main.py -data NSLKDD_global_0.1 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=0.1
python main.py -data NSLKDD_global_0.5 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=0.5
python main.py -data NSLKDD_global_1 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=1
python main.py -data NSLKDD_global_10 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=10

# CICIDS-2018 (Personalized test)
python main.py -data CICIDS_0.1 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=0.1
python main.py -data CICIDS_0.5 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=0.5
python main.py -data CICIDS_1 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=1
python main.py -data CICIDS_10 -m cicids -algo FedAvg -nc 5 -ncl 5 -topk 1 -fd 80 --skip_FL True #  β=10

# CICIDS-2018 (Global test)
python main.py -data CICIDS_global_0.1 -m cicids -algo FedAvg -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True #  β=0.1
python main.py -data CICIDS_global_0.5 -m cicids -algo FedAvg -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True #  β=0.5
python main.py -data CICIDS_global_1 -m cicids -algo FedAvg -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True #  β=1
python main.py -data CICIDS_global_10 -m cicids -algo FedAvg -nc 5 -ncl 7 -topk 1 -fd 80 --skip_FL True #  β=10

```
