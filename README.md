# CS5590_FoML
## Assignments on various datasets to understand the numerous ML algoriyhms worked on in association with the CS5590 coursework  in IIT Hyderabad


### [Assignment 1](https://github.com/JainAnki/CS5590_FoML/tree/main/BM21MTECH14001_Assign1/)
This homework is intended to cover programming exercises in the following topics:
- k-NN, Decision Trees, Model Selection, Naive Bayes classifier
- Determining the wine quality by implementing your own version of the decision tree using binary univariate split, entropy and information gain: 
   - https://archive.ics.uci.edu/ml/datasets/Wine+Quality - the Wine dataset

### [Assignment 2](https://github.com/JainAnki/CS5590_FoML/tree/main/BM21MTECH14001_Assign2/)
This homework is intended to cover theory and programming exercises in the following topics:
- SVM, Kernels
- soft-margin SVM to handwritten digits from the processed US Postal Service Zip Code data set: 
  - http://www.amlbook.com/data/zip/features.train 
  - http://www.amlbook.com/data/zip/features.test
- SVM to GISETTE is a handwritten digit recognition problem:
   - https://archive.ics.uci.edu/ml/datasets/Gisette
   - variations: In addition to the basic linear kernel, investigate two other standard kernels: RBF (a.k.a. Gaussian kernel), Polynomial kernel   

### [Assignment 3](https://github.com/JainAnki/CS5590_FoML/tree/main/BM21MTECH14001_Assign3/)
This homework is intended to cover programming exercises in the following topics:
- Neural Networks, Boosting/XGBoost, Random Forests
- Own random forest classifier (given you have written your own decision tree code) to apply to the Spam dataset 
   - data - https://hastie.su.domains/ElemStatLearn/datasets/spam.data
   - information - https://hastie.su.domains/ElemStatLearn/datasets/spam.info.txt
   - Explore the sensitivity of Random Forests to the parameter m (the number of features used for best split).
   - Plot the OOB (out-of-bag) error
- pre-processing methods and Gradient Boosting on the popular Lending Club dataset
   - Apply gradient boosting using the function sklearn.ensemble.GradientBoostingClassifier for training the model.
   - Get the best test accuracy you can, and show what hyperparameters led to this accuracy.
   - In particular, study the effect of increasing the number of trees in the classifier.
 
### [Assignment 4](https://github.com/JainAnki/CS5590_FoML/tree/main/BM21MTECH14001_Assign4/)
This homework is intended to cover theory and programming exercises in the following topics:
- Linear regression, Optimal bayes classifier, VC dimension, Regularizers
- Implement your own code for a logistic regression classifier
- Kaggle - Taxi Fare Prediction:
   - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction
   - model should achieve at least RMSE < 4.

### [Assignment 5](https://github.com/JainAnki/CS5590_FoML/tree/main/BM21MTECH14001_Assign5/)
- This homework is intended to cover programming exercises in the following topics:
   - Clustering, Expectation Maximization, PCA, t-SNE  
   - Implement your own DBSCAN algorithm
   - Use sklearn’s t-SNE algorithm and reduce the ’load digits’ from sklearn.datasets 64-dimensional data to 2 dimensions

### [Kaggle](https://github.com/JainAnki/CS5590_FoML/tree/main/BM21MTECH14001_BM21MTECH14004/)
- For this semester, we have created a Kaggle challenge named “Is the driver at fault?” exclusively for you. Here is the link: https://www.kaggle.com/t/53d1fd0d4cc8422bbdf9de2ea61f7e81. This challenge is designed based on the Driver Accident  tabular Dataset which is provided in the link. Kaggle is a crowd-sourced platform to attract, nurture, train and challenge data scientists from all around the world to solve data science, machine learning and predictive analytics problems. 

   A driver is at fault or no fault based on several direct and indirect variables specific to the accident and beyond.  The dataset has 42 features and almost all of them are categorical. Your aim is to train an ML algorithm that optimizes the evaluation metric i.e Accuracy. Train.csv file has the training data with Fault Label and input features. Test.csv file has unlabeled data. The detailed description of the challenge and dataset can be found in the aforementioned link. 
