import numpy
import pandas
import matplotlib.pyplot
import tensorflow

dftrain = pandas.read_csv("data/train.csv")
dfeval = pandas.read_csv("data/eval.csv")
y_train = dftrain.pop("survived")

categorical_columns = ["sex", "class", "deck", "embark_town", "alone"]
numeric_columns = ["age", "n_siblings_spouses", "parch", "fare"]

feature_columns = []

for feature in categorical_columns:
	vocabulary = dftrain[feature].unique()
	feature_columns.append(tensorflow.feature_column.categorical_column_with_vocabulary_list(feature,vocabulary))

for feature in numeric_columns:
	feature_columns.append(tensorflow.feature_column.numeric_column(feature, dtype = tensorflow.float32))
