
in a data set, all values that determine the final result, are called features

x = features
y = value/result

## feature engeneering
data should cover as many combinations of features as possible
you should aim for at least 10x more data points thatn features

* Add or drop features
 * choose the features that have the msot signal/importance

* combine multiple features into one feature
 * represent the data int he simplest way possible

* Binning
 * use the same measument values
 * represent numerical measurement with a more bread category

* One-hot encoding
 * process represent categorical data as numerical, but meaningfull

representing text to values, if posible like isX1 or isX2

Overfiting and underfiting

Overfitting
* Training set erro very low
* Test set error very high
* low dataset

Solution
* to make it less complex
 * smaller and simpler tree

Underfitting
* training set erro very high
* test set error very high

Solution: more complex model 
* More decision trees
* deeper 

Good fit
* training set erro low
* test set error low

