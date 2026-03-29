"""
customer-churn-analysis.py
===========================
End-to-end customer churn prediction pipeline built with Apache Spark MLlib.

This script orchestrates four ML stages:
  1. Data preprocessing and feature engineering
  2. Logistic Regression baseline model training and evaluation
  3. Chi-Square statistical feature selection
  4. Hyperparameter tuning and multi-model comparison using CrossValidator

Author: Vinay Devabhaktuni
Tech:   PySpark, Apache Spark MLlib, Python 3.8+
Run:    spark-submit customer-churn-analysis.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ---------------------------------------------------------------------------
# Setup: output directory and Spark session
# ---------------------------------------------------------------------------

# Create a local directory to persist all task outputs as text files
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Initialise the Spark session — entry point for all DataFrame and MLlib operations.
# 'CustomerChurnMLlib' is the application name visible in the Spark UI.
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

# Load the CSV dataset into a Spark DataFrame.
# header=True    : first row is the column header
# inferSchema=True: Spark automatically detects column data types
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Drop the customerID column — it is a unique identifier with no predictive value.
# Keeping it would introduce noise and risk data leakage.
df = df.drop("customerID")

# ---------------------------------------------------------------------------
# Task 1: Data Preprocessing and Feature Engineering
# ---------------------------------------------------------------------------

def preprocess_data(df):
    """
    Transform raw customer data into a feature-vector format suitable for MLlib.

    Steps performed:
      - Impute missing values in TotalCharges with 0 (customers with zero tenure
        have no accumulated charges, so 0 is semantically correct).
      - Encode categorical columns (gender, PhoneService, InternetService) using
        StringIndexer (label encoding) followed by OneHotEncoder (binary vectors).
      - Assemble all numeric and encoded features into a single 'features' vector
        using VectorAssembler — required by all Spark MLlib estimators.
      - Encode the target column 'Churn' (Yes/No) into a numeric label (1/0).

    Args:
        df: Raw Spark DataFrame loaded from customer_churn.csv

    Returns:
        Preprocessed Spark DataFrame with columns: ['features', 'label']
    """

    # --- Missing value imputation ---
    # TotalCharges can be null for new customers (tenure=0). Fill with 0 to
    # avoid propagating nulls through the pipeline.
    df = df.fillna({"TotalCharges": 0})

    # Categorical columns that need encoding before they can be used in ML models
    categorical_cols = ['gender', 'PhoneService', 'InternetService']

    # --- Categorical encoding: Step 1 - StringIndexer ---
    # Converts string category values to numeric indices based on frequency.
    # handleInvalid="keep" ensures unseen categories at inference time don't fail.
    # e.g., gender: Male -> 0, Female -> 1
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index", handleInvalid="keep") for col in categorical_cols]

    # --- Categorical encoding: Step 2 - OneHotEncoder ---
    # Converts integer indices into sparse binary vectors.
    # This prevents the model from treating category indices as ordinal values.
    # e.g., InternetService: DSL=[1,0], Fiber=[0,1], No=[0,0]
    encoders = [OneHotEncoder(inputCol=col + "_Index", outputCol=col + "_Vec") for col in categorical_cols]

    # Fit and transform each indexer then each encoder sequentially
    for stage in indexers + encoders:
        df = stage.fit(df).transform(df)

    # --- Feature assembly ---
    # VectorAssembler combines all feature columns into a single dense/sparse vector.
    # MLlib classifiers require a single 'features' column as input.
    # Feature order: [gender_Vec, PhoneService_Vec, InternetService_Vec,
    #                 SeniorCitizen, tenure, MonthlyCharges, TotalCharges]
    feature_cols = [col + "_Vec" for col in categorical_cols] + ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # --- Label encoding ---
    # StringIndexer converts the target column 'Churn' (Yes/No) to numeric:
    # 'No' -> 0.0 (majority class, assigned lowest index), 'Yes' -> 1.0
    df = StringIndexer(inputCol="Churn", outputCol="ChurnIndex").fit(df).transform(df)

    # Persist a sample of the preprocessed output for review and reporting
    with open(f"{output_dir}/task1_preprocessing_summary.txt", "w") as f:
        f.write("Task 1: Data Preprocessing and Feature Engineering\n")
        f.write("Sample Output:\n")
        # Collect 5 rows to Pandas for readable text output (safe at small scale)
        sample = df.select("features", "ChurnIndex").limit(5).toPandas()
        f.write(sample.to_string(index=False))

    # Return only the columns needed by downstream tasks.
    # Rename ChurnIndex -> label to match the MLlib convention for target columns.
    return df.select("features", "ChurnIndex").withColumnRenamed("ChurnIndex", "label")


# ---------------------------------------------------------------------------
# Task 2: Logistic Regression Baseline Model
# ---------------------------------------------------------------------------

def train_logistic_regression_model(df):
    """
    Train a Logistic Regression baseline and evaluate performance using AUC-ROC.

    Logistic Regression is chosen as the baseline because:
      - It is fast to train and interpretable
      - It provides a good linear boundary reference for comparison with
        tree-based and ensemble models in Task 4

    The model is evaluated using AUC-ROC (Area Under the ROC Curve):
      - AUC = 1.0 means perfect classification
      - AUC = 0.5 means no better than random guessing
      - For churn, AUC > 0.75 is generally considered acceptable

    Args:
        df: Preprocessed DataFrame with 'features' and 'label' columns
    """

    # Split into 80% training / 20% test.
    # seed=42 ensures the split is reproducible across runs.
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Train Logistic Regression with default parameters as a baseline.
    # Default: maxIter=100, regParam=0.0, elasticNetParam=0.0
    lr = LogisticRegression()
    model = lr.fit(train_df)

    # Generate predictions on the held-out test set
    predictions = model.transform(test_df)

    # Evaluate using Area Under the ROC Curve.
    # AUC-ROC is more robust than accuracy for imbalanced churn datasets.
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    # Persist results to file
    with open(f"{output_dir}/task2_logistic_regression_results.txt", "w") as f:
        f.write("Task 2: Logistic Regression Evaluation\n")
        f.write(f"Logistic Regression Model Accuracy (AUC): {auc:.2f}\n")


# ---------------------------------------------------------------------------
# Task 3: Feature Selection using Chi-Square Test
# ---------------------------------------------------------------------------

def feature_selection(df):
    """
    Select the top 5 most statistically predictive features using Chi-Square test.

    ChiSqSelector uses the chi-square test of independence to rank features
    by how strongly they depend on the target label. Features with higher
    chi-square scores are more likely to be informative for churn prediction.

    This reduces dimensionality, removes noise features, and can improve
    model generalisation on unseen data.

    Args:
        df: Preprocessed DataFrame with 'features' and 'label' columns
    """

    # Configure selector to pick the top 5 features from the assembled vector.
    # featuresCol and labelCol must match the column names used in preprocessing.
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    model = selector.fit(df)
    selected = model.transform(df)

    # Map vector indices back to human-readable feature names for reporting.
    # This is necessary because VectorAssembler merges all features into a single
    # indexed vector, losing the original column name mapping.
    categorical_cols = ['gender', 'PhoneService', 'InternetService']
    encoded_cols = [col + "_Vec" for col in categorical_cols]
    feature_names = encoded_cols + ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    # Determine total feature count from the first row's vector length
    sample_vector = df.select("features").first()[0]
    total_features = len(sample_vector)

    # Retrieve the indices of the selected features from the fitted model
    selected_indices = model.selectedFeatures
    selected_feature_names = []
    current_index = 0

    # Iterate through one-hot encoded columns (each expands to multiple binary features)
    for col in encoded_cols:
        for i in range(3):  # Each OHE column produces up to 3 binary indicator columns
            if current_index in selected_indices:
                selected_feature_names.append(f"{col}[{i}]")
            current_index += 1

    # Iterate through scalar numeric features (each occupies exactly 1 index)
    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        if current_index in selected_indices:
            selected_feature_names.append(col)
        current_index += 1

    # Persist selected feature names and sample reduced vectors to file
    with open(f"{output_dir}/task3_feature_selection.txt", "w") as f:
        f.write("Task 3: Feature Selection using Chi-Square\n")
        f.write("Top 5 features selected:\n")
        for name in selected_feature_names:
            f.write(f"- {name}\n")
        f.write("\nSample Output:\n")
        sample = selected.select("selectedFeatures", "label").limit(5).toPandas()
        f.write(sample.to_string(index=False))


# ---------------------------------------------------------------------------
# Task 4: Hyperparameter Tuning and Multi-Model Comparison
# ---------------------------------------------------------------------------

def tune_and_compare_models(df):
    """
    Tune and benchmark four classifiers using grid search with 5-fold cross-validation.

    Models evaluated:
      - Logistic Regression  : linear model, tuned on regularisation strength
      - Decision Tree        : tree-based model, tuned on maximum depth
      - Random Forest        : ensemble of trees, tuned on number of trees
      - Gradient Boosted Trees (GBT): sequential boosting, tuned on iterations

    Tuning strategy:
      - ParamGridBuilder defines the hyperparameter search space for each model
      - CrossValidator performs k-fold CV (k=5) to select the best parameter set
        without overfitting to the training data
      - Each model is evaluated on the same held-out test set for fair comparison

    AUC-ROC is used as the optimisation metric throughout.

    Args:
        df: Preprocessed DataFrame with 'features' and 'label' columns
    """

    # Use the same 80/20 split with seed=42 for fair comparison across all models
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Shared evaluator — AUC-ROC for binary classification
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    output_lines = ["Task 4: Hyperparameter Tuning and Model Comparison\n"]

    # Define each model alongside its hyperparameter grid.
    # ParamGridBuilder.addGrid(param, values) specifies the candidates to search.
    # CrossValidator will evaluate every combination via k-fold CV.
    models = {
        # Logistic Regression: tune L2 regularisation strength
        # Lower regParam -> less regularisation (risks overfitting)
        # Higher regParam -> stronger regularisation (risks underfitting)
        "LogisticRegression": (LogisticRegression(), ParamGridBuilder()
            .addGrid(LogisticRegression.regParam, [0.01, 0.1])
            .build()),

        # Decision Tree: tune maximum tree depth
        # Deeper trees capture more complex patterns but risk overfitting
        "DecisionTree": (DecisionTreeClassifier(), ParamGridBuilder()
            .addGrid(DecisionTreeClassifier.maxDepth, [5, 10])
            .build()),

        # Random Forest: tune number of trees in the ensemble
        # More trees generally improve accuracy but increase training time
        "RandomForest": (RandomForestClassifier(), ParamGridBuilder()
            .addGrid(RandomForestClassifier.numTrees, [10, 50])
            .build()),

        # Gradient Boosted Trees: tune number of boosting iterations
        # More iterations refine predictions but risk overfitting on noisy data
        "GBT": (GBTClassifier(), ParamGridBuilder()
            .addGrid(GBTClassifier.maxIter, [10, 20])
            .build())
    }

    # Train, tune, and evaluate each model
    for name, (model, grid) in models.items():
        output_lines.append(f"\nTuning {name}...")

        # CrossValidator wraps the estimator with k-fold cross-validation.
        # numFolds=5 means each parameter combination is evaluated 5 times,
        # each time using a different 80% training / 20% validation split.
        cv = CrossValidator(estimator=model,
                            estimatorParamMaps=grid,
                            evaluator=evaluator,
                            numFolds=5)

        # Fit the cross-validator — trains all parameter combinations on train_df
        cv_model = cv.fit(train_df)

        # Retrieve the model with the highest average validation AUC
        best_model = cv_model.bestModel

        # Evaluate the best model on the independent test set (never seen during CV)
        predictions = best_model.transform(test_df)
        auc = evaluator.evaluate(predictions)

        output_lines.append(f"{name} Best Model Accuracy (AUC): {auc:.2f}")
        output_lines.append(f"Best Params for {name}: {best_model.extractParamMap()}")

    # Persist all tuning results to a single output file
    with open(f"{output_dir}/task4_model_comparison.txt", "w") as f:
        f.write("\n".join(output_lines))


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

# Run all four pipeline stages sequentially.
# preprocessed_df is reused across tasks to avoid redundant computation.
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Cleanly shut down the Spark session to release cluster resources
spark.stop()
