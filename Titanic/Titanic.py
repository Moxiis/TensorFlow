import numpy
import pandas
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
import pyautogui
import time

time0 = time.time()
dftrain = pandas.read_csv("data/train.csv")
dfeval = pandas.read_csv("data/eval.csv")
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

categorical_columns = ["sex", "class", "deck", "embark_town", "alone"]
numeric_columns = ["age", "n_siblings_spouses", "parch", "fare"]

feature_columns = []

for feature in categorical_columns:
	vocabulary = dftrain[feature].unique()
	feature_columns.append(tensorflow.feature_column.categorical_column_with_vocabulary_list(feature,vocabulary))

for feature in numeric_columns:
	feature_columns.append(tensorflow.feature_column.numeric_column(feature, dtype = tensorflow.float32))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tensorflow.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tensorflow.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data
pyautogui.hotkey('ctrl', ',')

print("ACCURENCY:")
print(result['accuracy'])  # the result variable is simply a dict of stats about our model
print(" ")

person = 25
pred_dicts = list(linear_est.predict(eval_input_fn))
print("WHO:")
print(dfeval.loc[person])
print(" ")
print("SURVIVED:")
if y_eval[person] == 0:
	print("NO")
else:
	print("YES")
print(" ")
print("CHANCES ACCORDING TO THE MODEL")
print(pred_dicts[person]["probabilities"][1])


