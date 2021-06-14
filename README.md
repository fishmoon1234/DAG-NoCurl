# DAG-NoCurl

Code for DAG-NoCurl work

## Getting Started

### Prerequisites

```
Python 3.7
PyTorch >1.0
```


## How to Run 

Synthetic linear data experiments. Please download the dataset at
https://drive.google.com/file/d/1O52SlAHPRw_iFW_sAfm_vR3oMnoEb8am/view?usp=sharing

### Synthetic Experiments

CHOICE = nocurl, corresponding to the linear experiments, NoCurl-2 case in the paper
CHOICE = notear, corresponding to the linear experiments, NOTEARS case in the paper


```
python main_efficient.py --data_variable_size=10 --graph_type="erdos-renyi" --repeat=100 --methods=<CHOICE> --h_tol=1e-8 --graph_degree=4 --alpha_A=1000 --data_type="synthetic"

```


## Cite

If you make use of this code in your own work, please cite our paper:

```
@inproceedings{yu2021dag,
  title={DAGs with No Curl: An Efficient DAG Structure Learning Approach},
  author={Yue Yu, Tian Gao, Naiyu Yin and Qiang Ji},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  year={2021}
}
```


## Acknowledgments
Our work and code benefit from two existing works, which we are very grateful.

* DAG NOTEAR https://github.com/xunzheng/notears
* DAG NOFEAR https://github.com/skypea/DAG_No_Fear

