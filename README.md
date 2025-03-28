# Prototype Retrieval-Augmented Test-Time Optimization  for Federated Intrusion Detection System

All baselines and our method are implemented on top of [PFLlib](https://www.pfllib.com/benchmark.html). 
We are very grateful for this outstanding library.



## Dataset

All used datasets can be downloaded [here](https://drive.google.com/file/d/1mS2fbBCeXSvNeOlrvd0sOme2uUgTKqpJ/view?usp=sharing)

Please unzip the dataset.rar file into the dataset folder


## Quickly Start (example)
```
# NSLKDD (Personalized test)
python main.py -data NSLKDD -m cicids -algo ours -gr 128 -lbs 1024 -nc 5 -nb 5

# NSLKDD (Global test)
python main.py -data NSLKDD_global -m cicids -algo ours -gr 50 -lbs 128 -nc 5  -nb 5

# CICIDS2018 (Personalized test)
python main.py -data mini_cicids_2018 -m cicids -algo ours -gr 100 -lbs 1024 -nc 5 -nb 7

# CICIDS2018 (Global test)
python main.py -data mini_cicids_2018_global_test -m cicids -algo ours -gr 100 -lbs 1024 -nc 5 -nb 7

```
