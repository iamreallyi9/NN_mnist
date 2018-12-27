# coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path = './mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train = './data/mnist_train.tfrecords'
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test = './data/mnist_test.tfrecords'
data_path = './data'
resize_height = 28
resize_width = 28

#实现将标签和图片写成tfrecord 的功能
#第一步找到特征值和标签值
#第二步调用tf.train.Example.SerializeToString()
def write_tfRecord(tfRecordName, image_path, label_path):

    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0

    #Open file and return a stream【文件流】【f是一个fileObject】【‘r’指读取】
    #readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，列表可以由 for... in ... 结构进行处理
    #读取的是mnist_train_jpg_60000.txt文件，例如contents前三个['28755_0.jpg 0\n', '13360_5.jpg 5\n', '57662_5.jpg 5\n']
    #例如content=‘28755_0.jpg 0\n’时，value=['28755_0.jpg', '0']
    #image_path = ~~/
    #img_path = ~~/28755_0.jpg
    #使用tobytes()方法将图片转化为像素数字，得到的img_raw即为特征值x
    #标签值【结果值y_】得到的方法为生成10个数字的数组，将value列表里的标签值对应的数组里的位置置1,得到labels
    #【img_raw+labels】=feature(键值对形式)->Features ==>Example

    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split()
        img_path = image_path + value[0]
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1

        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        writer.write(example.SerializeToString())
        num_pic += 1
        print("the number of picture:", num_pic)
    writer.close()
    print("write tfrecord successful")

#生成tfrecord的函数，判断路径生成train和test两条记录
def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)



def read_tfRecord(tfRecord_path):
    # 该函数会生成一个先入出的队列，文件阅读器使用它来取据
    #shuffle=随机
    #serialized.example标量==解析==》》tensor# 张量img+labels
    #把读出的每个样本保存在 serialized_example中进行解序列化，标签和图片的键名应该和制作 tfrecords 的键名相同,其中标签给出几分类[10]
    #字符串转换为 8位无符号整型

    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([10], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label

#获取解析tfrecord后的图片batch和标签batch,调用read_tfRecord,isTrain来区分是否为训练集
#shuffle=随机，num_threasds=线程数
# capacity=队列中元素的最大数量，min_after_dequeue:=出队后列中的最小数量元素，=》用于确保的混合级别
def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test

    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=1000,
                                                    min_after_dequeue=700)
    return img_batch, label_batch


def main():
    #generate_tfRecord()
    print(get_tfrecord(10,True))


if __name__ == '__main__':
    main()
