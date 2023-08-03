
# # Import Libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, count, when
import numpy as np
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
import os 

# # Load Data
spark = SparkSession.builder.appName("MLlibTraining").getOrCreate()
print("SparkSession created.")
s3_path_train = sys.argv[1]
s3_path_val = sys.argv[2]

dir_path=os.path.dirname(s3_path_train)
output_csv_file = "output_log.csv"
s3_output_path = os.path.join(dir_path, output_csv_file)
model_dir = "model"
model_path = os.path.join(dir_path, model_dir)

df_train = spark.read.option("header", "true").option("delimiter", ";").option("quote", "\"").csv(s3_path_train)
df_val = spark.read.option("header", "true").option("delimiter", ";").option("quote", "\"").csv(s3_path_val)
df_train .show()
# create a SparkSession, read data from two CSV files located in Amazon S3, and display the content of the training dataset using the show()

# # Preprocess Data

# Count the number of null values in each column
null_counts = df_train.select([count(when(col(c).isNull(), c)).alias(c) for c in df_train.columns])
null_counts.show()
# There are no null values. So no need to handle null values


for col_name in df_train.columns:
    df_train = df_train.withColumn(col_name, col(col_name).cast(DoubleType()))
for col_name in df_val.columns:
    df_val = df_val.withColumn(col_name, col(col_name).cast(DoubleType()))

target_col = '""""quality"""""'#Define the target variable

(training_data, testing_data) = df_train, df_val# define training and testing sets

feature_columns = df_train.columns[:-1]# Extract feature columns (exclude the target variable column)
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
training_data = assembler.transform(training_data)
testing_data = assembler.transform(testing_data)


# **Data Type Conversion:**
# 
# The first part of the code converts all the columns in df_train and df_val to the DoubleType() data type. This is done to ensure that all the columns have numerical values, which is a requirement for machine learning algorithms in Spark.
# 
# **Define Target Variable:**
# 
# The code defines the target variable's name as '""""quality"""""'. This target variable is the variable we want to predict using machine learning models.
# 
# **Training and Testing Sets:**
# 
# The code then defines the training and testing datasets. It assigns the df_train to training_data and df_val to testing_data. These two DataFrames will be used for training and evaluating the machine learning models, respectively.
# 
# **Feature Columns:**
# 
# The code creates a list feature_columns, which includes all the columns in df_train except for the last one (last column is the target variable). These are the columns that will be used as features for the machine learning models.
# 
# **Vector Assembler:**
# 
# The VectorAssembler is used to combine the feature columns into a single vector column named "features". This is a required step in Spark MLlib as many algorithms expect the input features to be in a single vector column. The "features" column will be used as the input for the machine learning models.

# # Train Models


evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="weightedRecall")


# Create a DecisionTreeClassifier instance
dt = DecisionTreeClassifier(featuresCol="features", labelCol=target_col)

# Create a ParamGridBuilder with hyperparameters to tune
paramGrid = ParamGridBuilder()     .addGrid(dt.maxDepth, [5, 10, 15])     .addGrid(dt.maxBins, [32, 64, 128])     .build()

# Create a CrossValidator with the DecisionTreeClassifier, evaluator, and paramGrid
evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
crossval = CrossValidator(estimator=dt,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Run cross-validation to find the best hyperparameters
cvModel = crossval.fit(training_data)

# Get the best Decision Tree model from cross-validation
dt_model = cvModel.bestModel

dt_predictions = dt_model.transform(testing_data)
dt_accuracy = evaluator_accuracy.evaluate(dt_predictions)
dt_f1 = evaluator_f1.evaluate(dt_predictions)
dt_precision = evaluator_precision.evaluate(dt_predictions)
dt_recall = evaluator_recall.evaluate(dt_predictions)

print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Precision:", dt_precision)
print("Decision Tree F1:", dt_f1)
print("Decision Tree Recall :", dt_recall)
print()

# Create a RandomForestClassifier instance
rf = RandomForestClassifier(featuresCol="features", labelCol=target_col)

# Create a ParamGridBuilder with hyperparameters to tune
paramGrid = ParamGridBuilder()     .addGrid(rf.numTrees, [50, 100, 150])     .addGrid(rf.maxDepth, [5, 10, 15])     .build()

# Create a CrossValidator with the RandomForestClassifier, evaluator, and paramGrid
evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Run cross-validation to find the best hyperparameters
cvModel = crossval.fit(training_data)

# Get the best Random Forest model from cross-validation
rf_model = cvModel.bestModel

#Evaluate the Random Forest model
rf_predictions = rf_model.transform(testing_data)
rf_accuracy = evaluator_accuracy.evaluate(rf_predictions)
rf_f1 = evaluator_f1.evaluate(rf_predictions)
rf_precision = evaluator_precision.evaluate(rf_predictions)
rf_recall = evaluator_recall.evaluate(rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Precision:", rf_precision)
print("Random Forest F1:", rf_f1)
print("Random Forest Recall :", rf_recall)
print()                                   
                                      
lr = LogisticRegression(featuresCol="features", labelCol=target_col)

# Create a ParamGridBuilder with hyperparameters to tune
paramGrid = ParamGridBuilder()     .addGrid(lr.regParam, [0.01, 0.1, 0.3])     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])     .build()

# Create a CrossValidator with the LogisticRegression, evaluator, and paramGrid
evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# Run cross-validation to find the best hyperparameters
cvModel = crossval.fit(training_data)

# Get the best Logistic Regression model from cross-validation
lr_model = cvModel.bestModel

#Evaluate the Logistic Regression model
lr_predictions = lr_model.transform(testing_data)
lr_accuracy = evaluator_accuracy.evaluate(lr_predictions)
lr_f1 = evaluator_f1.evaluate(lr_predictions)
lr_precision = evaluator_precision.evaluate(lr_predictions)
lr_recall = evaluator_recall.evaluate(lr_predictions)
print("Logistic Regression Accuracy:", lr_accuracy)
print("Logistic Regression Precision:", lr_precision)
print("Logistic Regression F1:", lr_f1)
print("Logistic Regression Recall :", lr_recall)
print()                                    


# perform multi-class classification with three different classifiers: Decision Tree Classifier, Random Forest Classifier, and Logistic Regression Classifier. The data has already been prepared with the appropriate feature vectors and target variable.Also, this performe hyperparameter tuning for the Decision Tree Classifier, Random Forest Classifier, and Logistic Regression Classifier using cross-validation with different hyperparameter values. It evaluates each model's performance on the testing data using accuracy, precision, F1 score, and recall.


result_data = spark.createDataFrame([
    ("Decision Tree", dt_accuracy, dt_precision, dt_f1, dt_recall),
    ("Random Forest", rf_accuracy, rf_precision, rf_f1, rf_recall),
    ("Logistic Regression", lr_accuracy, lr_precision, lr_f1, lr_recall)
], ["Model", "Accuracy", "Precision", "F1 Score", "Recall"])


# Save the DataFrame to the CSV file in the S3 bucket
coalesced_df = result_data.coalesce(1)
coalesced_df.write.csv(s3_output_path, header=True, mode="overwrite")

print("Values have been written to:", s3_output_path)
# creates a DataFrame named result_data containing the evaluation results (accuracy, precision, F1 score, and recall) for the three classifiers: Decision Tree, Random Forest, and Logistic Regression. It then saves this DataFrame as a CSV file in the specified S3 bucket.

# # Save Model



max_=np.argmax(np.array((lr_accuracy,rf_accuracy,dt_accuracy)))
if max_==0:
    print('Logistic Regression saved')
    lr_model.write().overwrite().save(model_path)
elif max_==1:
    print('Random Forest saved')
    rf_model.write().overwrite().save(model_path)
elif max_==2:
    print('Decision Tree saved')
    dt_model.write().overwrite().save(model_path)


# determines which model (Logistic Regression, Random Forest, or Decision Tree) has the highest accuracy among the evaluated models and saves that best model to a specified path using the .save() method




