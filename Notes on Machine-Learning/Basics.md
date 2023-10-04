### What is Machine Learning ?

"Field of study that gives computers" the ability to learn without being explicitly programmed"

![[Pasted image 20231003152024.png]]

![[Pasted image 20231003152219.png]]

![[Pasted image 20231003152348.png]]


### Supervised Learning 

"Model able to predict with the help of labelled data"

![[Pasted image 20231003152537.png]]


### Importing Boston-data set

```python
import pandas as pd
from sklearn.dataset import load_boston
boston = load_boston()
dataset = pd.DataFrame(boston.data, columns = boston.feature_names)
```

### Importing Breast_cancer_dataset

```python 
import pandas as pd
from sklearn.dataset import load_breast_cancer
breast_cancer_data = load_breast_cancer()
dataset = pd.DataFrame(breast_cancer_data.data, columns = breast_cancer_data.feature_names)
dataset['target'] = breast_cancer_data['target']

scatter = dataset.plot(kind='scatter', x='mean radius', y='mean compactness', c='target', colormap='winter')

```

### Types of Machine Learning

**Supervised Learning:**
- Involves presenting labeled examples for the algorithm to learn from.
- Example: Predicting house selling prices using historical data.

**Unsupervised Learning:**
- Involves presenting examples without labels, requiring the algorithm to create labels.
- Example: Segmenting a customer database into similar groups based on characteristics and behaviors.

**Reinforcement Learning:**
- Involves presenting examples without labels, but receiving feedback from the environment.
- Used in settings like video games or stock markets where software learns from errors to find successful rules.


[[Linear models]]
[[Regression - analysis]]