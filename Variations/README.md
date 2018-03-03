# Toy neural network in numpy - variations

Buidling off the original toy neural network

# blobs.py

### New

![sklearn](http://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)

- [Generate isotropic Gaussian blobs]
- [Standardize features by removing the mean and scaling to unit variance]
- [Build a text report showing the main classification metrics]

![matplotlib](https://matplotlib.org/_static/logo2.png)

- [Make a scatter plot of x vs y]
- [Make a scatter plot of x vs neural network output]


[Generate isotropic Gaussian blobs]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

[Standardize features by removing the mean and scaling to unit variance]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

[Build a text report showing the main classification metrics]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

[Make a scatter plot of x vs y]: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter

[Make a scatter plot of x vs neural network output]: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter

### Summary

Randomly generated binary classification data set

100 samples, 50 per class, 3 features

Three hidden layers, tanh and sigmoid activation functions

### Plot of data set

![Figure_1](images/Figure_1.png) 

### Expected Output

![output_blobs](images/output_blobs.png)

![Figure_2](images/Figure_2.png)

# moons.py

### New

![sklearn](http://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)

- [Make two interleaving half circles]

[Make two interleaving half circles]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

### Summary

Interate through 3 different learning rates

Randomly generated binary classification data set

1000 samples, 500 per class, 2 features

Input Layer: 2 neurons

Hidden Layer 1,2,3: 8 neurons 

Three hidden layers, tanh and sigmoid activation functions

### Plot of data set

![moons](images/moons.png) 

### Output from different learning rates

![4](images/output_4.png) 
![1](images/1.png) 

![3](images/output_3.png) 
![01](images/01.png) 

![2](images/output_2.png) 
![001](images/001.png) 


