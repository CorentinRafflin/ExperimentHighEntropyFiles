
<h3>Forensics, project F3 </h3>
<h1>Experiments on high entropy files</h1>
<hr style="height:2px;border:none;color:#333;background-color:#333;"/>

_Author_
<div class="alert alert-warning">RAFFLIN Corentin </div>

<h2> 5) Classifier construction </h2>

The results of the experiments are saved in a file named `results.csv`, this notebook focuses on the processing of the data and the building of a classifier.

<div class="">
    <h3>1. Loading and treating the data</h3>
</div>


```python
#Diverses libraries
%matplotlib inline
import random
from time import time
import pickle
# Data and plotting imports
import pandas as pd
import numpy as np

#Neural network libraries
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid

#statistical libraries
from sklearn.preprocessing import LabelEncoder, RobustScaler 
from sklearn.model_selection import train_test_split# GridSearchCV,  KFold
from sklearn.svm import SVC
```

<h4> Loading the data </h4>


```python
#Path to the CSV file
resultsPath = 'results.csv'

#Header to associate to the CSV file
tests = ['File_type','File_bytes','Entropy','Chi_square','Mean','Monte_Carlo_Pi','Serial_Correlation'] 
cols = tests + [str(i) for i in range(0,256)]
```


```python
#Loading data
data = pd.read_csv(resultsPath, sep=',', header=None, names=cols)
print('There are {} files analyzed'.format(len(data)))
```

    There are 6220 files analyzed


<h4> Removing outliers and balancing the data </h4>


```python
countBefore = data['File_type'].value_counts().to_frame().rename(index=str, columns={'File_type':'Count_before'})

#Removing outliers by keeping only files with high entropy
data = data[data.Entropy>7.6]

countAfter = data['File_type'].value_counts().to_frame().rename(index=str, columns={'File_type':'Count_After'})

count = pd.concat([countBefore, countAfter], axis=1, sort=False)
display(count)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count_before</th>
      <th>Count_After</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pdf</th>
      <td>1613</td>
      <td>1140</td>
    </tr>
    <tr>
      <th>jpg</th>
      <td>1401</td>
      <td>1137</td>
    </tr>
    <tr>
      <th>png</th>
      <td>1136</td>
      <td>1103</td>
    </tr>
    <tr>
      <th>mp3</th>
      <td>1035</td>
      <td>1029</td>
    </tr>
    <tr>
      <th>zip</th>
      <td>1035</td>
      <td>1033</td>
    </tr>
  </tbody>
</table>
</div>



```python
#List of each file type
file_types = data['File_type'].sort_values().unique()

#List of dataframe for each file type 
files = [ data[data.File_type==file_type]  for file_type in file_types]

#Colors to associate to the file types
colors = ['r', 'b', 'g', 'y', 'm']

# In case more colors are needed for addition of other file type
'''
colors = list(pltcolors._colors_full_map.values())
random.seed(2)
random.shuffle(colors)
'''

print("File types :", file_types)
```

    File types : ['jpg' 'mp3' 'pdf' 'png' 'zip']



```python
#Removing some data (lower entropy) to have the same count for each file type
minCount = data['File_type'].value_counts().iloc[-1]
for i in range(len(files)):
    f = files[i]
    f = f.sort_values(by="Entropy")
    files[i] = f[len(f)-minCount:]

#Updating the full dataframe
data = pd.concat(files)
print('There are {} files analyzed'.format(len(data)))
```

    There are 5145 files analyzed


<h4> Checking for missing (possible errors) </h4>


```python
def getMissing(dataframe):
    ''' Printing the missing data in the dataframe with the total of missing and the corresponding percentage '''
    total = dataframe.isnull().sum().sort_values(ascending=False)
    percent = (dataframe.isnull().sum()/dataframe.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data[missing_data['Total']>0]
```


```python
#Checking for missing in the tests or bytes distribution
display(getMissing(data))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


No missing data in the tests which is great.

<div class="">
    <h3>2. Data Pre-processing</h3>
</div>

Now we will prepare the data for the clasifier.

<h4> Dropping and separating input-ouput </h4>


```python
#Dropping not useful information 
data = data.drop('File_bytes', axis=1)

#Separating the output
y = data['File_type']
data = data.drop('File_type', axis=1)
```

<h4> Splitting into training and testing sets </h4>


```python
#Splitting into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size = 0.1, random_state=7)
```

<h4> Standardization </h4>

The <b> standardization </b> of a dataset is a common requirement for many machine learning estimators. We use the RobustScaler more robust to outliers as it is possible that we have many outliers in this data set. The centering and scaling statistics of this scaler are based on percentiles and are therefore not influenced by a few number of very large marginal outliers. The outliers themselves are still present in the transformed data. 

> Typically this is done by removing the mean and scaling to unit variance. However, outliers can often influence the sample mean / variance in a negative way. In such cases, the median and the interquartile range often give better results.


```python
#Scaling features using statistics that are robust to outliers.
scaler = RobustScaler() 
#Fitting on the training set, then transforming both training and testing sets
X_train = scaler.fit(X_train).transform(X_train)
X_test = scaler.transform(X_test)
```

There is no need for dimensionality reduction nor for decorrelating the data using PCA.

<h4> Encoding the output </h4>


```python
lbencoder = LabelEncoder()
lbencoder.fit(Y_train)
Y_train = lbencoder.transform(Y_train)
Y_test = lbencoder.transform(Y_test)
```


```python
#Printing the shapes
print("Shape x_train", X_train.shape)
print("Shape y_train", Y_train.shape)
print("Shape x_test", X_test.shape)
print("Shape y_test", Y_test.shape)
```

    Shape x_train (4630, 261)
    Shape y_train (4630,)
    Shape x_test (515, 261)
    Shape y_test (515,)


<div class="">
    <h3>3. Model Selection</h3>
</div>

Several classifiers could be used for this problem. I decided to focus on SVM which is good for limited data and a Neural Network (Multi Layer Perceptron Classifier) which is fast for prediction and therefore could be better to implement in the `ent` program.

>In short:
* Boosting - often effective when a large amount of training data is available.
* Random trees - often very effective and can also perform regression.
* K-nearest neighbors - simplest thing you can do, often effective but slow and requires lots of memory.
* Neural networks - Slow to train but very fast to run, still optimal performer for letter recognition.
* SVM - Among the best with limited data, but losing against boosting or random trees only when large data sets are available.
https://stackoverflow.com/questions/2595176/which-machine-learning-classifier-to-choose-in-general

In this part I will not focus on the optimization of the parameters.

I used the f1_score as a metric to give weights to all classes and see the accuracy for each class.  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

<h4> a. MultiLayer Perceptron (MLP) </h4>

> MLPs are suitable for classification prediction problems where inputs are assigned a class or label. They are very flexible and can be used generally to learn a mapping from inputs to outputs.


```python
mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(50), random_state=1, max_iter=50000, activation='relu', 
                    learning_rate_init=0.00001, verbose=False)
```


```python
mlp.fit(X_train, Y_train)
```




    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=50, learning_rate='constant',
           learning_rate_init=1e-05, max_iter=50000, momentum=0.9,
           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
           random_state=1, shuffle=True, solver='adam', tol=0.0001,
           validation_fraction=0.1, verbose=False, warm_start=False)




```python
#Results 
print("Training set score: %f" % mlp.score(X_train, Y_train))
print("Testing set score: %f" % mlp.score(X_test, Y_test))
Y_pred = mlp.predict(X_test)
print("F1 test set score:", metrics.f1_score(Y_test, Y_pred , average=None))
print("Corresponding classes:", lbencoder.classes_)
print("F1 mean test set score:", metrics.f1_score(Y_test, Y_pred , average='macro'))
```

    Training set score: 0.982721
    Testing set score: 0.941748
    F1 test set score: [0.95789474 1.         0.98039216 0.87700535 0.87628866]
    Corresponding classes: ['jpg' 'mp3' 'pdf' 'png' 'zip']
    F1 mean test set score: 0.9383161802184494


<h4> b. SVC </h4>

> SVC and NuSVC implement the “one-against-one” approach (Knerr et al., 1990) for multi- class classification. If n_class is the number of classes, then n_class * (n_class - 1) / 2 classifiers are constructed and each one trains data from two classes. 
https://scikit-learn.org/stable/modules/svm.html

A one against one approach may be better to differentiate two close classes like zip and png.



```python
svc = SVC(gamma='scale', decision_function_shape='ovo')
```


```python
svc.fit(X_train, Y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
#Results
print("Training set score: %f" % svc.score(X_train, Y_train))
print("Testing set score: %f" % svc.score(X_test, Y_test))
Y_pred = svc.predict(X_test)
print("F1 test set score:", metrics.f1_score(Y_test, Y_pred , average=None))
print("Corresponding classes:", lbencoder.classes_)
print("F1 mean test set score:", metrics.f1_score(Y_test, Y_pred , average='macro'))
```

    Training set score: 0.964579
    Testing set score: 0.953398
    F1 test set score: [0.97916667 1.         0.98007968 0.89617486 0.9       ]
    Corresponding classes: ['jpg' 'mp3' 'pdf' 'png' 'zip']
    F1 mean test set score: 0.951084242265909


<div class="">
    <h3>4. Parameter Optimisation</h3>
</div>

<h4> a. MultiLayer Perceptron (MLP) </h4>

I kept the activation function `relu` which from experience and theory gives good results.


```python
# Define the hyperparameters
hyperparameters = {
    'solver': ['adam', 'lbfgs'], 
    'hidden_layer_sizes' : [(20), (50), (100), (10,10)], 
    'lr' : [0.00005, 0.0001]
}

# Compute all combinations
parameter_grid = list(ParameterGrid(hyperparameters))

# Just a table to save the results
resultsDF = pd.DataFrame(columns=['solver', 'hidden_layer_sizes', 'lr', 'test_score', 'train_score', 'f1_mean'])

for p in parameter_grid:   
    mlp = MLPClassifier(solver=p['solver'],hidden_layer_sizes=p['hidden_layer_sizes'], random_state=1, max_iter=50000,
                        activation='relu', learning_rate_init=p['lr'], early_stopping=True, tol=1e-7 )
    mlp.fit(X_train, Y_train)
    
    test_score = mlp.score(X_test, Y_test)    
    p['test_score'] = test_score
    
    train_score = mlp.score(X_train, Y_train)    
    p['train_score'] = train_score
    
    Y_pred = mlp.predict(X_test)
    f1_mean = metrics.f1_score(Y_test, Y_pred , average='macro')
    p['f1_mean']=f1_mean
    
    resultsDF = resultsDF.append(p, ignore_index=True)
    
display(resultsDF.sort_values('test_score', ascending=False).head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>solver</th>
      <th>hidden_layer_sizes</th>
      <th>lr</th>
      <th>test_score</th>
      <th>train_score</th>
      <th>f1_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>lbfgs</td>
      <td>50</td>
      <td>0.00005</td>
      <td>0.957282</td>
      <td>1.0</td>
      <td>0.954074</td>
    </tr>
    <tr>
      <th>7</th>
      <td>lbfgs</td>
      <td>50</td>
      <td>0.00010</td>
      <td>0.957282</td>
      <td>1.0</td>
      <td>0.954074</td>
    </tr>
    <tr>
      <th>9</th>
      <td>lbfgs</td>
      <td>100</td>
      <td>0.00005</td>
      <td>0.953398</td>
      <td>1.0</td>
      <td>0.950701</td>
    </tr>
    <tr>
      <th>11</th>
      <td>lbfgs</td>
      <td>100</td>
      <td>0.00010</td>
      <td>0.953398</td>
      <td>1.0</td>
      <td>0.950701</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lbfgs</td>
      <td>20</td>
      <td>0.00005</td>
      <td>0.939806</td>
      <td>1.0</td>
      <td>0.935572</td>
    </tr>
  </tbody>
</table>
</div>


<h4> b. SVC </h4>


```python
# Define the hyperparameters
hyperparameters = {
    'C':[0.5, 1, 5],
    'kernel':['rbf', 'linear', 'poly']
}

# Compute all combinations
parameter_grid = list(ParameterGrid(hyperparameters))

# Just a table to save the results
resultsDF = pd.DataFrame(columns=['C', 'kernel', 'test_score', 'train_score', 'f1_mean'])

for p in parameter_grid:   
    svc = SVC(gamma='scale', C=p['C'], kernel=p['kernel'], decision_function_shape='ovo')
    svc.fit(X_train, Y_train)
    
    test_score = svc.score(X_test, Y_test)    
    p['test_score'] = test_score
    
    train_score = svc.score(X_train, Y_train)    
    p['train_score'] = train_score
    
    Y_pred = svc.predict(X_test)
    f1_mean = metrics.f1_score(Y_test, Y_pred , average='macro')
    p['f1_mean']=f1_mean
    
    resultsDF = resultsDF.append(p, ignore_index=True)
    
display(resultsDF.sort_values('test_score', ascending=False).head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>kernel</th>
      <th>test_score</th>
      <th>train_score</th>
      <th>f1_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>5.0</td>
      <td>rbf</td>
      <td>0.955340</td>
      <td>0.989201</td>
      <td>0.952933</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>rbf</td>
      <td>0.953398</td>
      <td>0.964579</td>
      <td>0.951084</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.5</td>
      <td>rbf</td>
      <td>0.951456</td>
      <td>0.948596</td>
      <td>0.949330</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.5</td>
      <td>linear</td>
      <td>0.939806</td>
      <td>0.988769</td>
      <td>0.935264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>linear</td>
      <td>0.926214</td>
      <td>0.992441</td>
      <td>0.920596</td>
    </tr>
  </tbody>
</table>
</div>


<div class="">
    <h3>5. Final Model </h3>
</div>

Though there are not big differences of accuracy for these two models, MLP classifier has slighlty better score.


```python
clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(50), random_state=1, max_iter=60000, activation='relu', 
                learning_rate_init=0.000005, early_stopping=True, tol=1e-7)
clf.fit(X_train, Y_train)
```




    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=True, epsilon=1e-08,
           hidden_layer_sizes=50, learning_rate='constant',
           learning_rate_init=5e-06, max_iter=60000, momentum=0.9,
           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
           random_state=1, shuffle=True, solver='lbfgs', tol=1e-07,
           validation_fraction=0.1, verbose=False, warm_start=False)




```python
#Results
print("Training set score: %f" % clf.score(X_train, Y_train))
print("Testing set score: %f" % clf.score(X_test, Y_test))
Y_pred = clf.predict(X_test)
f1_score = metrics.f1_score(Y_test, Y_pred, average=None)
Y_pred = clf.predict(X_test)
print("F1 test set score:", metrics.f1_score(Y_test, Y_pred , average=None))
print("Corresponding classes:", lbencoder.classes_)
print("F1 mean test set score:", metrics.f1_score(Y_test, Y_pred , average='macro'))
```

    Training set score: 1.000000
    Testing set score: 0.957282
    F1 test set score: [0.95789474 1.         0.99224806 0.91397849 0.90625   ]
    Corresponding classes: ['jpg' 'mp3' 'pdf' 'png' 'zip']
    F1 mean test set score: 0.954074258696253


We noticed that the mp3 class is always correctly predicted, and almost always for pdf. It was to be expected with the distributions insofar as most of their distributions were distinct from the others.  
Similarly, we notice that png and zip have the lowest accuracy which is certainly due to the fact that they are difficult to distinguish and therefore the classifier may make mistakes between these two classes.

There is a bit of overfitting as the testing set accuracy does not reach a perfect accuracy such as the training set. We would need to increase the number of samples for each class to improve the accuracy on the testing set.

If we added more classes, it is likely that the global accuracy would lower due to similarity between some file types like we have for png and zip.

<h4> Saving </h4>


```python
filename = "scaler_lb_clf.sav"
modlist = [scaler, lbencoder, clf]
s = pickle.dump(modlist, open(filename, 'wb'))
```
