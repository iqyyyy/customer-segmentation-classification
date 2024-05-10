# Customer Segmentation Classification

Classification of product purchase data by customers to determine future customer segments using Random Forest, XGBoost, Logistic Regression, and Neural Network algorithms.


### Section

- [1 - Import library](#1)
- [2 - Data Wrangling](#2)
- [3 - Exploratory data analysis](#3)
    - [3.1 - Explore the variables](#3.1)
    - [3.2 - Explore product category](#3.2)
    - [3.3 - Explore customer category using K-Means Algorithm](#3.3)
- [4 - Classification Customer](#4)
    - [4.1 - Random forest](#4.1)
    - [4.2 - XGboost](#4.2)
    - [4.3 - Logistic Regression](#4.3)
    - [4.4 - Deep Neural Network](#4.4)
- [5 - Testing the predictions](#5)
- [6 - Summary](#6)

### Result

From the results of our tests, the best model is the Voting Classifier which combines models from 3 algorithms namely Random Forest, XGBoost, and Logistic Regression.

Overall, this dataset has features that can be explored more deeply, and the use of other models can also be applied to this case. The limitation of this dataset is that the time span is only one year, so it is not possible to perform time series analysis for forecasting.

We have clustered the products into 7 clusters using the Unsupervised Learning algorithm KMeans, and using KMeans we have also categorized the customers into 10 clusters. After clustering we do classification to see how suitable the given category is for each customer.

Classification is done in the last 2 months, the algorithm made successfully classifies correctly ~88% of the given class, this is a relatively high and good number and does not cause overfiting or underfiting.


### Setup environment with Anaconda
```
conda create --n class
conda activate class
conda install numpy pandas matplotlib seaborn jupyter nltk sklearn tensorflow
```

### with Python
```
conda install numpy pandas matplotlib seaborn jupyter nltk sklearn tensorflow
```
