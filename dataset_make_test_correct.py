import tensorflow as tf
import time
import forward
import dataset_make_backward
import dataset_make
import os

TEST_IN_SEC = 5
TEST_NUM = 10000

def test_co():
    #测试首先要复现模型，也就是需要想，xyy_,w，b【w,b就要考虑滑动平均值】

    #个人理解：会话之前的内容都是描述这个graph的结构，包括输入输出，学习方式，优化方法等，但并不进行实际运算
    #然后在sess里，进行一定次数的运算
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[None,forward.IN_NODE])
        y_ = tf.placeholder(tf.float32,[None,forward.OUT_NODE])
        y = forward.forward(x,None)

        ema = tf.train.ExponentialMovingAverage(dataset_make_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
        accuary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        img_batch,label_batch =dataset_make.get_tfrecord(TEST_NUM,False)

        #计算正确率，不需要轮数，就是一直在计算，global_step也是一直从别的文件读进来的
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(dataset_make_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    coord = tf.train.Coordinator()  # 3
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 4
                    xs,ys =  sess.run([img_batch,label_batch])
                    accuary_score = sess.run(accuary,feed_dict={x:xs,y_:ys})

                    print("轮数为 %s 准确率为 %s "%(global_step,accuary_score))

                    coord.request_stop()  # 6
                    coord.join(threads)  # 7
                else:
                    print("error!!!@@@")
                    return
            time.sleep(TEST_IN_SEC)


def main():

    test_co()


if __name__ == '__main__':
    main()
