import pandas as pd
import tensorflow as tf
import numpy as np
import random
BATCH_SIZE = 50
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 5000
VALIDATION_SIZE = 2000
SUMMARY_DIR = "/log/supervisor.log"
train = pd.read_csv('./Data/train.csv').as_matrix().astype(np.uint8)
test = pd.read_csv('./Data/test.csv').as_matrix().astype(np.uint8)
train_images = train[:-VALIDATION_SIZE, :]
validation = train_images[:VALIDATION_SIZE, :]
validation_image = validation[:,1:]
validation_image = np.multiply(validation_image,1.0/255.0)
validation_label = validation[:,0]
test = np.multiply(test, 1.0 / 255.0)
sess = tf.InteractiveSession()
def minibatch(train_images,batch_size):
    batch = random.sample(list(train_images), batch_size)
    batch = np.array(batch)
    images = batch[:, 1:]
    images = np.multiply(images, 1.0 / 255.0)
    labels = batch[:, 0]
    return images, labels
def getrandomvalidation(validation,batch_size):
    batch = random.sample(list(validation), batch_size)
    batch = np.array(batch)
    batch_validation_image = batch[:,1:]
    batch_validation_image = np.multiply(batch_validation_image,1.0/255.0)
    batch_validation_label = batch[:,0]
    return batch_validation_image, batch_validation_label
def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def label2vec(label, numclass):
    num_labels = label.shape[0]
    index  = np.arange(num_labels)*numclass
    label_vec = np.zeros((num_labels,numclass))
    label_vec.flat[index+label.ravel()] = 1
    return label_vec

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])
keep_prob = tf.placeholder(tf.float32)

w_conv1 = weight_variable([5,5,1,32])
tf.summary.histogram('w_conv1', w_conv1)
b_conv1 = bias_variable([32])
tf.summary.histogram('b_conv1', b_conv1)
conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
conv1_pool = max_pool(conv1)

w_conv2 = weight_variable([5,5,32,64])
tf.summary.histogram('w_conv2', w_conv2)
b_conv2 = bias_variable([64])
tf.summary.histogram('w_conv2', w_conv2)
conv2 = tf.nn.relu(conv2d(conv1_pool,w_conv2)+b_conv2)
conv2_pool = max_pool(conv2)

w_fc1= weight_variable([7*7*64,1024])
tf.summary.histogram('w_fc1',w_fc1)
b_fc1 = bias_variable([1024])
tf.summary.histogram('b_fc1',b_fc1)
fc1_flat = tf.reshape(conv2_pool,[-1,7*7*64])
fc1 = tf.nn.relu(tf.matmul(fc1_flat,w_fc1)+b_fc1)
fc1_drop = tf.nn.dropout(fc1,keep_prob)

w_fc2 = weight_variable([1024,10])
tf.summary.histogram('w_fc2',w_fc2)
b_fc2 = bias_variable([10])
tf.summary.histogram('w_fc2',w_fc2)
y_conv = tf.matmul(fc1_drop,w_fc2)+b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
tf.summary.scalar('cross entropy',cross_entropy)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy',accuracy)
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
saver = tf.train.Saver()
tf.global_variables_initializer().run()
for i in range(TRAINING_EPOCHS):
    images, labels = minibatch(train,BATCH_SIZE)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: images, y_: label2vec(labels,10),keep_prob: 1.0} )
        batch_validation_image, batch_validation_label = getrandomvalidation(validation, BATCH_SIZE)
        validation_accuracy = accuracy.eval(feed_dict={
            x: batch_validation_image, y_: label2vec(batch_validation_label,10),keep_prob: 1.0})
        print("step %d, training accuracy %.2f  => validation_accuracy %.2f" % (i, train_accuracy, validation_accuracy))
    train_step.run(feed_dict={x: images, y_: label2vec(labels,10), keep_prob: 0.8})
    summary = sess.run(merged,feed_dict={x: images, y_: label2vec(labels,10), keep_prob: 0.8})
    summary_writer.add_summary(summary,i)
saver.save(sess,'./save/model.ckpt')
all_validation_accuracy = accuracy.eval(feed_dict={
            x: validation_image, y_: label2vec(validation_label,10),keep_prob: 1.0})
print('all_validation_accuracy => %.4f'%all_validation_accuracy)
summary_writer.close()
prediction = tf.argmax(y_conv, 1)
predicted_lables = np.zeros(test.shape[0])
for i in range(0,test.shape[0]//BATCH_SIZE):
    predicted_lables[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = prediction.eval(
        feed_dict={x: test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], keep_prob: 1.0})
np.savetxt('submission_norm.csv',
           np.c_[range(1,len(test)+1),predicted_lables],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')
