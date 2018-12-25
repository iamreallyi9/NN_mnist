import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#minst = input_data.read_data_sets('./m_data',one_hot = True)

#打印数据集信息
#print (minst.train.num_examples)
#print(minst.validation.num_examples)
#print(minst.test.num_examples)
#print(minst.train.labels[0])
#print(minst.train.images[0])

#BATCH_SIZE = 200
#随机抽取出BATCH_SIZE个信息进行训练
#xs, ys = minst.train.next_batch(BATCH_SIZE)
#print(xs,ys.shape)



## prepare the original data
with tf.name_scope('data'):
     x_data = np.random.rand(100).astype(np.float32)
     y_data = 0.3*x_data+0.1
##creat parameters
with tf.name_scope('parameters'):
     weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
     bias = tf.Variable(tf.zeros([1]))
##get y_prediction
with tf.name_scope('y_prediction'):
     y_prediction = weight*x_data+bias
##compute the loss
with tf.name_scope('loss'):
     loss = tf.reduce_mean(tf.square(y_data-y_prediction))
##creat optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
#creat train ,minimize the loss
with tf.name_scope('train'):
     train = optimizer.minimize(loss)
#creat init
with tf.name_scope('init'):
     init = tf.global_variables_initializer()
##creat a Session
sess = tf.Session()
##initialize
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    if step %10==0 :
        print (step ,'weight:',sess.run(weight),'bias:',sess.run(bias))

