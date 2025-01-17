# MXNet2Caffe: Convert MXNet model to Caffe model

You are welcome to file issues either for bugs in the source code, feature requests!

## fit to python3

forked from https://github.com/cypw/MXNet2Caffe and fit it to python3

## Brief Guide

Given a MXNet Model, MXNet2Caffe can automatically generate both `.prototxt` and `.caffemodel`.

Before starting using MXNet2Caffe, you need to manually set the path in `find_caffe.py` and `find_mxnet.py`.

After that, simply run `python json2prototxt.py` to generate the corresponding `.prototxt`.

And then, using `python mxnet2caffe.py` to generate the corresponding `.caffemodel`.


## TODO

[1] Since there is no `Flatten` layer in caffe, you have to manually modify the automatically generated `.prototxt`. In other words, you have to change the `bottom` of the layer just after the `Flatten` layer making it linking to the layer before the `Flatten` layer. Currently, this part has to be done manually.

[2] The converted model scores a little bit worse than the original MXNet model.

[3] Code for automatically reversing the weight (and bias) of the first layer to support BGR input.

[4] Better support for caffe's in-place feature.

[5] Several TODOs in `prototxt_basic.py`
