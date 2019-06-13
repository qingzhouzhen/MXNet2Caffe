caffe_root='/data0/hhq/project/caffe-master/'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe
import numpy as np
import struct
import matplotlib.pyplot as plt
import cv2
import os

# caffe.set_mode_gpu()
# caffe.set_device(1)

lables = ['black', 'green', 'red']

deploy='model_caffe/ped_light_classify_night.prototxt'
caffe_model= 'model_caffe/ped_light_classify_night.caffemodel'

def predict_all():
    # path = 'pic/B1000_1.jpg'
    # path = 'pic/black/190.121.11.119_20180101_080207_00124_1.jpg' # black
    # path = 'pic/green/aa2600000005_1.jpg' # green
    # path = 'pic/yellow/wuhe_zhangheng_S_34_9-00_9-30-033_0.jpg'
    # path = "pic/red/aa1700000017_0.jpg"
    # path = '/data0/hhq/data/light/test/yellow/190.201.11.101_20180101_075246_00231_1_0.jpg'
    # path = 'pic/yellow/huaian-qingyuan-W-2-20181219130000-20181219140000-344962234_24_1_0.jpg'
    path = 'pic/yellow/0-0-light-classify-resize-tmp.jpg'
    # path = 'pic/yellow/0-0-red140269870044672.jpg'
    # root_path = '/data0/hhq/project/light-classify/data/buble_label/'
    root_path = '/data0/hhq/project/light-classify/data/light_6_1/test/'
    # root_path = '/data0/hhq/project/light-classify/data/test/'
    first_direction_list = os.listdir(root_path)

    net = caffe.Net(deploy, caffe_model, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    # transformer.set_channel_swap('data', (2, 1, 0))  # should not exchange
    net.blobs['data'].reshape(1, 3, 128, 64)

    for first_directin in first_direction_list:
        abs_first_direction = os.path.join(root_path, first_directin)
        if not os.path.isdir(abs_first_direction): continue
        file_name_list = os.listdir(abs_first_direction)
        for file_name in file_name_list:
            abs_file_name = os.path.join(abs_first_direction, file_name)

            im = caffe.io.load_image(abs_file_name)

            image_pro = transformer.preprocess('data', im)

            net.blobs['data'].data[...] = image_pro

            out = net.forward()

            fc = net.blobs['fc'].data[0].flatten()
            conv1 = net.blobs['conv_1_conv2d'].data[0]
            conv1_bn = net.blobs['conv_1_batchnorm'].data[0]
            prob = net.blobs['prob'].data[0].flatten()
            # print(fc)
            # print(prob)
            order = prob.argsort()[2]

            print('the class is:', order, lables[order], "                    ", abs_file_name)

def predict_single():
    # image_path ='pic/red/aa1700000017_0.jpg' # red

    # image_path = 'pic/yellow/wuhe_zhangheng_S_34_9-00_9-30-033_0.jpg' # yellow

    # image_path ='pic/black/190.121.11.119_20180101_080207_00134_0.jpg' # black

    # image_path = 'pic/green/aa2600000006_0.jpg'
    # image_path = 'pic/yellow/0-0-light-classify-resize-tmp.jpg'
    # image_path = 'pic/yellow/0-0-red140269870044672.jpg'
    image_path = '/data0/hhq/project/light-classify/data/mid.jpg'
    image_path = "/data0/hhq/project/light-classify/pic/0-0-ped-light-classify-crop.jpg"

    net = caffe.Net(deploy, caffe_model, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 225)
    # transformer.set_channel_swap('data', (2, 1, 0))  # should not exchange
    net.blobs['data'].reshape(1, 3, 128, 64)

    im = caffe.io.load_image(image_path)

    image_pro = transformer.preprocess('data', im)

    net.blobs['data'].data[...] = image_pro

    out = net.forward()

    fc = net.blobs['fc'].data[0].flatten()
    conv1 = net.blobs['conv_1_conv2d'].data[0]
    conv1_bn = net.blobs['conv_1_batchnorm'].data[0]
    prob = net.blobs['prob'].data[0].flatten()
    print(fc)
    print(prob)
    order = prob.argsort()[2]
    print(order)

    print('the class is:', order, lables[order])

predict_single()
# predict_all()
