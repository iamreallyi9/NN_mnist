import tensorflow as tf
import numpy as np
from PIL import Image
import os


def write_tfRecord(tfRecordName,image_path,label_path):
    #A class to write records to a TFRecords file
    #将image 和label 转化为tfrecord
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path,'r')
    contents = f.readline()
    for content in contents:
        value = content.split()










def main():
    pass

if __name__ == '__main__':
    main()
