import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df = df.drop("customerID")

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    df = df.fillna({"TotalCharges": 0})
    categorical_cols = ['gender', 'PhoneService', 'InternetService']
    
    # Index and encode categorical features
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index", handleInvalid="keep") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "_Index", outputCol=col + "_Vec") for col in categorical_cols]

    for stage in indexers + encoders:
        df = stage.fit(df).transform(df)

    # Assemble features
    feature_cols = [col + "_Vec" for col in categorical_cols] + ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # Label encoding
    df = StringIndexer(inputCol="Churn", outputCol="ChurnIndex").fit(df).transform(df)

    # Save output sample
    with open(f"{output_dir}/task1_preprocessing_summary.txt", "w") as f:
        f.write("Task 1: Data Preprocessing and Feature Engineering\n")
        f.write("Sample Output:\n")
        sample = df.select("features", "ChurnIndex").limit(5).toPandas()
        f.write(sample.to_string(index=False))

    return df.select("features", "ChurnIndex").withColumnRenamed("ChurnIndex", "label")

# Task 2: Train and Evaluate Logistic Regression Model
def train_logistic_regression_model(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression()
    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    with open(f"{output_dir}/task2_logistic_regression_results.txt", "w") as f:
        f.write("Task 2: Logistic Regression Evaluation\n")
        f.write(f"Logistic Regression Model Accuracy (AUC): {auc:.2f}\n")

# Task 3: Feature Selection using Chi-Square
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    model = selector.fit(df)
    selected = model.transform(df)

    # Map feature indices to approximate names
    categorical_cols = ['gender', 'PhoneService', 'InternetService']
    encoded_cols = [col + "_Vec" for col in categorical_cols]
    feature_names = encoded_cols + ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    sample_vector = df.select("features").first()[0]
    total_features = len(sample_vector)

    selected_indices = model.selectedFeatures
    selected_feature_names = []
    current_index = 0
    for col in encoded_cols:
        for i in range(3):  # assume 3 one-hot values
            if current_index in selected_indices:
                selected_feature_names.append(f"{col}[{i}]")
            current_index += 1
    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
        if current_index in selected_indices:
            selected_feature_names.append(col)
        current_index += 1

    with open(f"{output_dir}/task3_feature_selection.txt", "w") as f:
        f.write("Task 3: Feature Selection using Chi-Square\n")
        f.write("Top 5 features selected:\n")
        for name in selected_feature_names:
            f.write(f"- {name}\n")
        f.write("\nSample Output:\n")
        sample = selected.select("selectedFeatures", "label").limit(5).toPandas()
        f.write(sample.to_string(index=False))

# Task 4: Hyperparameter Tuning and Model Comparison
def tune_and_compare_models(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    output_lines = ["Task 4: Hyperparameter Tuning and Model Comparison\n"]

    models = {
        "LogisticRegression": (LogisticRegression(), ParamGridBuilder()
                               .addGrid(LogisticRegression.regParam, [0.01, 0.1])
                               .build()),
        "DecisionTree": (DecisionTreeClassifier(), ParamGridBuilder()
                         .addGrid(DecisionTreeClassifier.maxDepth, [5, 10])
                         .build()),
        "RandomForest": (RandomForestClassifier(), ParamGridBuilder()
                         .addGrid(RandomForestClassifier.numTrees, [10, 50])
                         .build()),
        "GBT": (GBTClassifier(), ParamGridBuilder()
                .addGrid(GBTClassifier.maxIter, [10, 20])
                .build())
    }

    for name, (model, grid) in models.items():
        output_lines.append(f"\nTuning {name}...")
        cv = CrossValidator(estimator=model,
                            estimatorParamMaps=grid,
                            evaluator=evaluator,
                            numFolds=5)
        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel
        predictions = best_model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        output_lines.append(f"{name} Best Model Accuracy (AUC): {auc:.2f}")
        output_lines.append(f"Best Params for {name}: {best_model.extractParamMap()}")

    with open(f"{output_dir}/task4_model_comparison.txt", "w") as f:
        f.write("\n".join(output_lines))

# Run all tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()