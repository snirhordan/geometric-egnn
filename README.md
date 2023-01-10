# geometric-egnn



### QM9 experiment
properties --> [alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve]  
learning rate --> 1e-3 for [gap, homo lumo], 5r-4 for the rest
```
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_1_alpha
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property gap --exp_name exp_1_gap
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property homo --exp_name exp_1_homo
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property lumo --exp_name exp_1_lumo
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property mu --exp_name exp_1_mu
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property Cv --exp_name exp_1_Cv
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property G --exp_name exp_1_G
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property H --exp_name exp_1_H
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property r2 --exp_name exp_1_r2
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property U --exp_name exp_1_U
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property U0 --exp_name exp_1_U0
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property zpve --exp_name exp_1_zpve
```
