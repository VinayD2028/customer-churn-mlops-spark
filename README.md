# handson-10-MachineLearning-with-MLlib.

#  Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

---



Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

##  Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

##  Tasks

### Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.

**Code Output:**

```
+--------------------+-----------+
|features            |ChurnIndex |
+--------------------+-----------+
|[0.0,12.0,29.85,29...|0.0        |
|[0.0,1.0,56.95,56....|1.0        |
|[1.0,5.0,53.85,108...|0.0        |
|[0.0,2.0,42.30,184...|1.0        |
|[0.0,8.0,70.70,151...|0.0        |
+--------------------+-----------+
```
---

### Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Use `BinaryClassificationEvaluator` to evaluate.

**Code Output Example:**
```
Logistic Regression Model Accuracy: 0.83
```

---

###  Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Print the selected feature vectors.

**Code Output Example:**
```
+--------------------+-----------+
|selectedFeatures    |ChurnIndex |
+--------------------+-----------+
|[0.0,29.85,0.0,0.0...|0.0        |
|[1.0,56.95,1.0,0.0...|1.0        |
|[0.0,53.85,0.0,1.0...|0.0        |
|[1.0,42.30,0.0,0.0...|1.0        |
|[0.0,70.70,0.0,1.0...|0.0        |
+--------------------+-----------+

```

---

### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use CrossValidator to tune models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` for 5-fold cross-validation.
3. Evaluate and print best model results.

**Code Output Example:**
```
Tuning LogisticRegression...
LogisticRegression Best Model Accuracy (AUC): 0.84
Best Params for LogisticRegression: regParam=0.01, maxIter=20

Tuning DecisionTree...
DecisionTree Best Model Accuracy (AUC): 0.77
Best Params for DecisionTree: maxDepth=10

Tuning RandomForest...
RandomForest Best Model Accuracy (AUC): 0.86
Best Params for RandomForest: maxDepth=15
numTrees=50

Tuning GBT...
GBT Best Model Accuracy (AUC): 0.88
Best Params for GBT: maxDepth=10
maxIter=20

```
---

##  Execution Instructions

### 1. Prerequisites

- Apache Spark installed
- Python environment with `pyspark` installed
  ```bash
  pip install pyspark
  ```
- `customer_churn.csv` placed in the project directory

### 2. Run the Project
1. generate the input dataset:
```bash
python3 dataset-generator.py 
```
2. Execute the tasks:
```bash
spark-submit customer-churn-analysis.py 
```
---
###  original ouput and explaination of the code
---
## Task 1: Data Preprocessing and Feature Engineering
**Objective:** Transform raw data into machine-readable format for modeling.

### Key Steps:
1. **Handle Missing Values**
   - Fills null values in `TotalCharges` with 0 using `df.fillna()`.
   
2. **Categorical Feature Encoding**
   - **String Indexing:** Converts text categories (`gender`, `PhoneService`, `InternetService`) to numeric indices.
   - **One-Hot Encoding:** Converts numeric indices to binary vectors (e.g., `[1.0, 0.0]` for "Fiber" internet).

3. **Feature Assembly**
   - Combines encoded categorical features and numeric features (`SeniorCitizen`, `tenure`, etc.) into a single feature vector using `VectorAssembler`.

4. **Label Encoding**
   - Converts target column `Churn` (Yes/No) to numeric labels (1/0) using `StringIndexer`.

### Output:
- Saves sample preprocessed data to `outputs/task1_preprocessing_summary.txt`.
```bash
Task 1: Data Preprocessing and Feature Engineering
Sample Output:
                                                      features  ChurnIndex
  [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 60.82, 121.24]         1.0
  [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 7.0, 77.35, 521.82]         0.0
 (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 12.0, 20.19, 132.71)         0.0
[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 33.0, 94.45, 2815.79]         1.0
   [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 7.0, 83.24, 646.6]         1.0
```
---

## Task 2: Logistic Regression Baseline
**Objective:** Establish baseline performance for churn prediction.

### Key Steps:
1. **Data Splitting**
   - Splits data into 80% training / 20% test sets with fixed random seed (`seed=42`).

2. **Model Training**
   - Trains logistic regression using default parameters (`LogisticRegression()`).

3. **Evaluation**
   - Calculates **AUC-ROC** (Area Under ROC Curve) using `BinaryClassificationEvaluator`.

### Output:
- Saves AUC score to `outputs/task2_logistic_regression_results.txt`.
```bash
Task 2: Logistic Regression Evaluation
Logistic Regression Model Accuracy (AUC): 0.73
```
---

## Task 3: Feature Selection
**Objective:** Identify top 5 most predictive features using Chi-Square test.

### Key Steps:
1. **Chi-Square Selector**
   - Uses `ChiSqSelector` to select features with the highest statistical dependence on the target.

2. **Feature Mapping**
   - Maps selected vector indices back to approximate original feature names:
     - One-hot encoded features: `gender_Vec`, `InternetService_Vec`.
     - Numeric features: `tenure`, `MonthlyCharges`.

### Output:
- Saves selected feature names and sample output to `outputs/task3_feature_selection.txt`.
```bash
Task 3: Feature Selection using Chi-Square
Top 5 features selected:
- PhoneService_Vec[1]
- PhoneService_Vec[2]
- InternetService_Vec[0]
- InternetService_Vec[1]
- InternetService_Vec[2]

Sample Output:
          selectedFeatures  label
 [0.0, 1.0, 0.0, 1.0, 2.0]    1.0
 [1.0, 0.0, 0.0, 1.0, 7.0]    0.0
(0.0, 0.0, 1.0, 0.0, 12.0)    0.0
[0.0, 1.0, 0.0, 1.0, 33.0]    1.0
 [0.0, 1.0, 0.0, 1.0, 7.0]    1.0
```
---

## Task 4: Model Tuning & Comparison
**Objective:** Optimize multiple models and identify the best performer.

### Models Evaluated:
1. **Logistic Regression**
   - Tuned Parameters: Regularization (`regParam`), Iterations (`maxIter`).

2. **Decision Tree**
   - Tuned Parameter: Maximum depth (`maxDepth`).

3. **Random Forest**
   - Tuned Parameters: Number of trees (`numTrees`).

4. **Gradient Boosted Trees (GBT)**
   - Tuned Parameters: Boosting iterations (`maxIter`).

### Key Steps:
1. **Grid Search**
   - Tests parameter combinations using `ParamGridBuilder`.

2. **5-Fold Cross-Validation**
   - Reduces overfitting risk through cross-validation.

3. **Model Selection**
   - Selects parameters with the best validation AUC.

4. **Final Evaluation**
   - Reports test set AUC for each model.

### Output:
- Saves tuning results to `outputs/task4_model_comparison.txt`.
```bash
Task 4: Hyperparameter Tuning and Model Comparison


Tuning LogisticRegression...
LogisticRegression Best Model Accuracy (AUC): 0.73
Best Params for LogisticRegression: {Param(parent='LogisticRegression_af2b6604af4f', name='aggregationDepth', doc='suggested depth for treeAggregate (>= 2).'): 2, Param(parent='LogisticRegression_af2b6604af4f', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.'): 0.0, Param(parent='LogisticRegression_af2b6604af4f', name='family', doc='The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial'): 'auto', Param(parent='LogisticRegression_af2b6604af4f', name='featuresCol', doc='features column name.'): 'features', Param(parent='LogisticRegression_af2b6604af4f', name='fitIntercept', doc='whether to fit an intercept term.'): True, Param(parent='LogisticRegression_af2b6604af4f', name='labelCol', doc='label column name.'): 'label', Param(parent='LogisticRegression_af2b6604af4f', name='maxBlockSizeInMB', doc='maximum memory in MB for stacking input data into blocks. Data is stacked within partitions. If more than remaining data size in a partition then it is adjusted to the data size. Default 0.0 represents choosing optimal value, depends on specific algorithm. Must be >= 0.'): 0.0, Param(parent='LogisticRegression_af2b6604af4f', name='maxIter', doc='max number of iterations (>= 0).'): 100, Param(parent='LogisticRegression_af2b6604af4f', name='predictionCol', doc='prediction column name.'): 'prediction', Param(parent='LogisticRegression_af2b6604af4f', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.'): 'probability', Param(parent='LogisticRegression_af2b6604af4f', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'): 'rawPrediction', Param(parent='LogisticRegression_af2b6604af4f', name='regParam', doc='regularization parameter (>= 0).'): 0.0, Param(parent='LogisticRegression_af2b6604af4f', name='standardization', doc='whether to standardize the training features before fitting the model.'): True, Param(parent='LogisticRegression_af2b6604af4f', name='threshold', doc='Threshold in binary classification prediction, in range [0, 1]. If threshold and thresholds are both set, they must match.e.g. if threshold is p, then thresholds must be equal to [1-p, p].'): 0.5, Param(parent='LogisticRegression_af2b6604af4f', name='tol', doc='the convergence tolerance for iterative algorithms (>= 0).'): 1e-06}

Tuning DecisionTree...
DecisionTree Best Model Accuracy (AUC): 0.66
Best Params for DecisionTree: {Param(parent='DecisionTreeClassifier_40796c1bbe35', name='cacheNodeIds', doc='If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval.'): False, Param(parent='DecisionTreeClassifier_40796c1bbe35', name='checkpointInterval', doc='set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext.'): 10, Param(parent='DecisionTreeClassifier_40796c1bbe35', name='featuresCol', doc='features column name.'): 'features', Param(parent='DecisionTreeClassifier_40796c1bbe35', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini', Param(parent='DecisionTreeClassifier_40796c1bbe35', name='labelCol', doc='label column name.'): 'label', Param(parent='DecisionTreeClassifier_40796c1bbe35', name='leafCol', doc='Leaf indices column name. Predicted leaf index of each instance in each tree by preorder.'): '', Param(parent='DecisionTreeClassifier_40796c1bbe35', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32, Param(parent='DecisionTreeClassifier_40796c1bbe35', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30].'): 5, Param(parent='DecisionTreeClassifier_40796c1bbe35', name='maxMemoryInMB', doc='Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size.'): 256, Param(parent='DecisionTreeClassifier_40796c1bbe35', name='minInfoGain', doc='Minimum information gain for a split to be considered at a tree node.'): 0.0, Param(parent='DecisionTreeClassifier_40796c1bbe35', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 1, Param(parent='DecisionTreeClassifier_40796c1bbe35', name='minWeightFractionPerNode', doc='Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5).'): 0.0, Param(parent='DecisionTreeClassifier_40796c1bbe35', name='predictionCol', doc='prediction column name.'): 'prediction', Param(parent='DecisionTreeClassifier_40796c1bbe35', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.'): 'probability', Param(parent='DecisionTreeClassifier_40796c1bbe35', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'): 'rawPrediction', Param(parent='DecisionTreeClassifier_40796c1bbe35', name='seed', doc='random seed.'): -1441058006940296601}

Tuning RandomForest...
RandomForest Best Model Accuracy (AUC): 0.79
Best Params for RandomForest: {Param(parent='RandomForestClassifier_18f18ba42d0a', name='bootstrap', doc='Whether bootstrap samples are used when building trees.'): True, Param(parent='RandomForestClassifier_18f18ba42d0a', name='cacheNodeIds', doc='If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval.'): False, Param(parent='RandomForestClassifier_18f18ba42d0a', name='checkpointInterval', doc='set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext.'): 10, Param(parent='RandomForestClassifier_18f18ba42d0a', name='featureSubsetStrategy', doc="The number of features to consider for splits at each tree node. Supported options: 'auto' (choose automatically for task: If numTrees == 1, set to 'all'. If numTrees > 1 (forest), set to 'sqrt' for classification and to 'onethird' for regression), 'all' (use all features), 'onethird' (use 1/3 of the features), 'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)), 'n' (when n is in the range (0, 1.0], use n * number of features. When n is in the range (1, number of features), use n features). default = 'auto'"): 'auto', Param(parent='RandomForestClassifier_18f18ba42d0a', name='featuresCol', doc='features column name.'): 'features', Param(parent='RandomForestClassifier_18f18ba42d0a', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini'): 'gini', Param(parent='RandomForestClassifier_18f18ba42d0a', name='labelCol', doc='label column name.'): 'label', Param(parent='RandomForestClassifier_18f18ba42d0a', name='leafCol', doc='Leaf indices column name. Predicted leaf index of each instance in each tree by preorder.'): '', Param(parent='RandomForestClassifier_18f18ba42d0a', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32, Param(parent='RandomForestClassifier_18f18ba42d0a', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30].'): 5, Param(parent='RandomForestClassifier_18f18ba42d0a', name='maxMemoryInMB', doc='Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size.'): 256, Param(parent='RandomForestClassifier_18f18ba42d0a', name='minInfoGain', doc='Minimum information gain for a split to be considered at a tree node.'): 0.0, Param(parent='RandomForestClassifier_18f18ba42d0a', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 1, Param(parent='RandomForestClassifier_18f18ba42d0a', name='minWeightFractionPerNode', doc='Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5).'): 0.0, Param(parent='RandomForestClassifier_18f18ba42d0a', name='numTrees', doc='Number of trees to train (>= 1).'): 20, Param(parent='RandomForestClassifier_18f18ba42d0a', name='predictionCol', doc='prediction column name.'): 'prediction', Param(parent='RandomForestClassifier_18f18ba42d0a', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.'): 'probability', Param(parent='RandomForestClassifier_18f18ba42d0a', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'): 'rawPrediction', Param(parent='RandomForestClassifier_18f18ba42d0a', name='seed', doc='random seed.'): -7387420455837441889, Param(parent='RandomForestClassifier_18f18ba42d0a', name='subsamplingRate', doc='Fraction of the training data used for learning each decision tree, in range (0, 1].'): 1.0}

Tuning GBT...
GBT Best Model Accuracy (AUC): 0.74
Best Params for GBT: {Param(parent='GBTClassifier_b13bc2ff1497', name='cacheNodeIds', doc='If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval.'): False, Param(parent='GBTClassifier_b13bc2ff1497', name='checkpointInterval', doc='set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext.'): 10, Param(parent='GBTClassifier_b13bc2ff1497', name='featureSubsetStrategy', doc="The number of features to consider for splits at each tree node. Supported options: 'auto' (choose automatically for task: If numTrees == 1, set to 'all'. If numTrees > 1 (forest), set to 'sqrt' for classification and to 'onethird' for regression), 'all' (use all features), 'onethird' (use 1/3 of the features), 'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)), 'n' (when n is in the range (0, 1.0], use n * number of features. When n is in the range (1, number of features), use n features). default = 'auto'"): 'all', Param(parent='GBTClassifier_b13bc2ff1497', name='featuresCol', doc='features column name.'): 'features', Param(parent='GBTClassifier_b13bc2ff1497', name='impurity', doc='Criterion used for information gain calculation (case-insensitive). Supported options: variance'): 'variance', Param(parent='GBTClassifier_b13bc2ff1497', name='labelCol', doc='label column name.'): 'label', Param(parent='GBTClassifier_b13bc2ff1497', name='leafCol', doc='Leaf indices column name. Predicted leaf index of each instance in each tree by preorder.'): '', Param(parent='GBTClassifier_b13bc2ff1497', name='lossType', doc='Loss function which GBT tries to minimize (case-insensitive). Supported options: logistic'): 'logistic', Param(parent='GBTClassifier_b13bc2ff1497', name='maxBins', doc='Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature.'): 32, Param(parent='GBTClassifier_b13bc2ff1497', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30].'): 5, Param(parent='GBTClassifier_b13bc2ff1497', name='maxIter', doc='max number of iterations (>= 0).'): 20, Param(parent='GBTClassifier_b13bc2ff1497', name='maxMemoryInMB', doc='Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size.'): 256, Param(parent='GBTClassifier_b13bc2ff1497', name='minInfoGain', doc='Minimum information gain for a split to be considered at a tree node.'): 0.0, Param(parent='GBTClassifier_b13bc2ff1497', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 1, Param(parent='GBTClassifier_b13bc2ff1497', name='minWeightFractionPerNode', doc='Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5).'): 0.0, Param(parent='GBTClassifier_b13bc2ff1497', name='predictionCol', doc='prediction column name.'): 'prediction', Param(parent='GBTClassifier_b13bc2ff1497', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.'): 'probability', Param(parent='GBTClassifier_b13bc2ff1497', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'): 'rawPrediction', Param(parent='GBTClassifier_b13bc2ff1497', name='seed', doc='random seed.'): -8262332687822256406, Param(parent='GBTClassifier_b13bc2ff1497', name='stepSize', doc='Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator.'): 0.1, Param(parent='GBTClassifier_b13bc2ff1497', name='subsamplingRate', doc='Fraction of the training data used for learning each decision tree, in range (0, 1].'): 1.0, Param(parent='GBTClassifier_b13bc2ff1497', name='validationTol', doc='Threshold for stopping early when fit with validation is used. If the error rate on the validation input changes by less than the validationTol, then learning will stop early (before `maxIter`). This parameter is ignored when fit without validation is used.'): 0.01}
```
---

## Pipeline Workflow
**Steps:**  
1. Data Preparation →  
2. Baseline Model →  
3. Feature Importance →  
4. Model Optimization  

**Final Output:**  
Text files in the `outputs/` directory containing:
- Preprocessing samples.
- Baseline AUC score.
- Top 5 features.
- Tuned model parameters and performance.
