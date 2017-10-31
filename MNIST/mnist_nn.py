import pandas as pd
import tensorflow as tf
import numpy as np
train = pd.read_csv('./Data/train.csv').as_matrix().astype(np.uint8)
test = pd.read_csv('./Data/test.csv').as_matrix().astype(np.uint8)
print(train.shape)
print(test.shape)
y_train = train['label']
X_train = train.drop('label',1)
X_test = test

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1,name=name)
    return tf.Variable(initial)
def label2vec(label, numclass):
    num_labels = label.shape[0]
    index  = np.arange(num_labels)*numclass
    label_vec = np.zeros(num_labels,numclass)
    label_vec.flat[index+label.ravel()] = 1
    return label_vec

