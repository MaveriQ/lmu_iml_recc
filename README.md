<div align="center">    
 
# Interpreting Neural Network based Recommender Systems  (Work in Progress)   

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/MaveriQ/lmu_iml_recc/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
We are using Netflix dataset to build a simple Recommender System in PyTorch. Then we use various NN interpretability methods to find insights into the patterns learning by the NN.   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/MaveriQ/lmu_iml_recc

# install project   
cd lmu_iml_recc 
conda env create -f environment.yml
conda activate lmu_iml_recc
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd models

# run module (example: mnist as your main contribution)   
python pl_netflix.py    
```

<!-- 
## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer
# model
model = LitClassifier()
# data
train, val, test = mnist()
# train
trainer = Trainer()
trainer.fit(model, train, val)
# test using the best model!
trainer.test(test_dataloaders=test)
```
### Citation   
```
@article{Haris Jabbar,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
``` --> 
