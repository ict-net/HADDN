# HADDN
Source code for ToN'21 paper: "[Interactive Anomaly Detection in Dynamic Communication Networks](https://ieeexplore.ieee.org/abstract/document/9494106/)".
# Requirements
- python: 3.9
- numpy: 1.25.0
- pandas: 2.0.2
- matplotlib: 3.7.2
- scikit-learn: 1.3.0
- seaborn: 0.12.0
- tqdm: 4.65.0

# Usage
### Reproduce our results of UCB_HADDN
1. Run the code with sample data extracted from CICIDS2017.
```
python3 ucb_haddn.py
```
2. Labeling results of the time period ```t``` and ```t+1``` can be found in ```./ucb_old.csv``` and ```./ucb_new.csv```, and the anomaly detection results of the time period ```t+1``` can be found in ```./ucb_test.csv```.
### Reproduce our results of TS_HADDN
1. Run the code
```
python3 ts_haddn.py
```
2. Labeling results of the time period ```t``` and ```t+1``` can be found in ```./ts_old.csv``` and ```./ts_new.csv```, and the anomaly detection results of the time period ```t+1``` can be found in ```./ts_test.csv```.
# Others
Please cite our paper if you use this code in your own work:
```
@article{MengWWYZ21,
  title        = {Interactive Anomaly Detection in Dynamic Communication Networks},
  author       = {Xuying Meng and
                  Yequan Wang and
                  Suhang Wang and
                  Di Yao and
                  Yujun Zhang},
  journal      = {{IEEE/ACM} Trans. Netw.},
  volume       = {29},
  number       = {6},
  pages        = {2602--2615},
  year         = {2021}
}
```