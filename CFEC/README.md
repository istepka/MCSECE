![Tests](https://github.com/LoGosX/counterfactuals/actions/workflows/tests.yml/badge.svg) ![flake8,mypy](https://github.com/LoGosX/counterfactuals/actions/workflows/code_analysis.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/counterfactuals/badge/?version=latest)](https://counterfactuals.readthedocs.io/en/latest/?badge=latest)
# CFEC

This is a specialized programming library which contains three different counterfactual generation methods for tabular data, supporting various constraints, and to construct a tool for comparing their results.

## Requirements
Tha package has been tested under python 3.7 up to 3.9, on both Windows and Ubuntu platforms. Its main dependency is tensorflow, which all the methods use, and typical scientific stack (numpy, scipy, pandas).
Requirements include:
* tensorflow~=2.7.0
* pandas==1.3.4
* numpy==1.21.4
* scikit-learn==1.0.1



## Installation
This package can be installed using pip
```bash
pip install cfec
```

## Implemented algorithms
Our package includes implementation of algorithms, such as: 
* [FIMAP](https://ojs.aaai.org/index.php/AAAI/article/view/17362)
* [CADEX](https://doi.org/10.1007/978-3-030-29908-8\_4)
* [Ensemble](https://arxiv.org/abs/2102.13076)

## Example usage
```python
from cfec.explainers import Fimap
from cfec.constraints import ValueMonotonicity, ValueNominal
from data import AdultData
from sklearn.ensemble import RandomForestClassifier

adult_data = AdultData('data/datasets/adult.csv')

rf = RandomForestClassifier()
rf.fit(adult_data.X_train, adult_data.y_train)

predictions = rf.predict(adult_data.X_train)

constraints = [
    OneHot('workclass', 2, 8),
    OneHot('martial.status', 9, 15),
    OneHot('occupation', 16, 29),
    OneHot('race', 30, 34),
    OneHot('sex', 35, 36),
]

fimap = Fimap(constraints=constraints)

fimap.fit(adult_data.X_train, predictions)

x = adult_data.X_train.iloc[0]
cf = fimap.generate(x)
```

```python
from cfec.explainers import Cadex
from cfec.constraints import ValueMonotonicity, ValueNominal
from data import GermanData
from tensorflow import keras

german_data = GermanData('data/datasets/input_german.csv', 'data/datasets/labels_german.csv')

# simple model consisting of one dense layer with 2 units and a softmax activation
german_model = keras.models.load_model('models/model_german')

predictions = german_model.predict(german_data.X_train)

constraints = [
    OneHot("account_status", 7, 10), 
    OneHot("credit_history", 11, 15),
    OneHot("purpose", 16, 25), 
    OneHot("savings", 26, 30), 
    OneHot("sex_status", 31, 34),
    OneHot("debtors", 35, 37), 
    OneHot("property", 38, 41),
    OneHot("other_installment_plans", 42, 44), 
    OneHot("housing", 45, 47), 
    OneHot("job", 48, 51),
    OneHot("phone", 52, 53), 
    OneHot("foreign", 54, 55), 
    OneHot("employment", 56, 60)
]

cadex = Cadex(german_model, constraints=constraints)

x = german_data.X_train.iloc[0]
cf = cadex.generate(x) # cadex method does not need to fit before generate
```