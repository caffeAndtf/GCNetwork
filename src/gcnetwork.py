from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, Conv3D, Conv2DTranspose
from conv3dTranspose import Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras import backend as K
from keras.layers import Input, Add, add, multiply
from keras.layers.core import Lambda, Permute, Reshape
import tensorflow as tf


# only build layer structures, no data flow
def _resNetBlock_(filters, ksize, stride, padding, act_func):
    conv1 = Conv2D(filters, ksize, strides=stride, padding=padding)
    bn1 = BatchNormalization(axis=-1)
    act1 = Activation(act_func)
    conv2 = Conv2D(filters, ksize, strides=stride, padding=padding)
    bn2 = BatchNormalization(axis=-1)
    act2 = Activation(act_func)
    add = Add()
    return [conv1, bn1, act1, conv2, bn2, act2, add]


def _addConv3D_(input, filters, ksize, stride, padding, bn=True, act_func='relu'):
    conv = Conv3D(filters, ksize, strides=stride, padding=padding)(input)
    if bn:
        conv = BatchNormalization(axis=-1)(conv)
    if act_func:
        conv = Activation(act_func)(conv)
    return conv


def _convDownSampling_(input, filters, ksize, ds_stride, padding):
    conv = _addConv3D_(input, filters, ksize, ds_stride, padding)
    conv = _addConv3D_(conv, filters, ksize, 1, padding)    
    conv = _addConv3D_(conv, filters, ksize, 1, padding)    
    return conv


def _createDeconv3D_(input, filters, ksize, stride, padding, bn=True, act_func='relu'):
    deconv = Conv3DTranspose(filters, ksize, stride, padding)(input)
    if bn:
        deconv = BatchNormalization(axis=-1)(deconv)
    if act_func:
        deconv = Activation(act_func)(deconv)
    return deconv


def _highwayBlock_(tensor):
    output, input, trans = tensor
    return add([multiply([output, trans]), multiply([input, 1 - trans])])


def _getCostVolume_(inputs, max_d):
    left_tensor, right_tensor = inputs
    shape = K.shape(right_tensor) #(batch, height, width, channel)
    right_tensor = K.spatial_2d_padding(right_tensor, padding=((0, 0), (max_d, 0)))
    disparity_costs = []
    for d in reversed(range(max_d)):
        left_tensor_slice = left_tensor
        right_tensor_slice = tf.slice(right_tensor, begin=[0, 0, d, 0], size=[-1, -1, shape[2], -1])
        cost = K.concatenate([left_tensor_slice, right_tensor_slice], axis=3)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1) #(batch, D, height, width, 2*channel)
    return cost_volume


def _computeLinearScore_(cv, d):
    cv = K.permute_dimensions(cv, (0, 2, 3, 1))
    disp_map = K.reshape(K.arange(0, d, dtype=K.floatx()), (1, 1, d, 1))
    output = K.conv2d(cv, disp_map, strides=(1, 1), padding='valid')
    return K.squeeze(output, axis=-1)


def _computeSoftArgMin_(cv, d):
    softmax = tf.nn.softmax(cv, dim=1)
    #softmax = K.permute_dimensions(softmax, (0,2,3,1))
    disp_map = K.reshape(K.arange(0, d, dtype='float32'), (1, 1, d, 1))
    output = K.conv2d(softmax, disp_map, strides=(1, 1), data_format='channels_first', padding='valid')
    return K.squeeze(output, axis=1)


def getOutputFunction(output):
    if output == 'linear':
        return _computeLinearScore_
    if output == 'softargmin':
        return _computeSoftArgMin_


def _createUniFeature_(input_shape, num_res, filters, first_ksize, ksize, act_func, ds_stride, padding):
    conv1 = Conv2D(filters, first_ksize, strides=ds_stride, padding=padding, input_shape=input_shape)
    bn1 = BatchNormalization(axis=-1)
    act1 = Activation(act_func)
    layers = [conv1, bn1, act1]
    for i in range(num_res):
        layers += _resNetBlock_(filters, ksize, 1, padding, act_func)
    output = Conv2D(filters, ksize, strides=1, padding=padding)
    layers.append(output)
    return layers


def _LearnReg_(input, base_num_filters, ksize, ds_stride, resnet, padding, highway_func, num_down_conv):    
    down_convs = list()
    conv = _addConv3D_(input, base_num_filters, ksize, 1, padding) 
    conv = _addConv3D_(conv, base_num_filters, ksize, 1, padding)
    down_convs.insert(0, conv)
    if not resnet:
        trans_gates = list()
        gate = _addConv3D_(conv, base_num_filters, ksize, 1, padding)
        trans_gates.insert(0, gate)
    for i in range(num_down_conv):
        if i < num_down_conv - 1:
            mult = 2
        else:
            mult = 4
        conv = _convDownSampling_(conv, mult * base_num_filters, ksize, ds_stride, padding)
        down_convs.insert(0, conv)
        if not resnet:
            gate = _addConv3D_(conv, mult * base_num_filters, ksize, 1, padding)
            trans_gates.insert(0, gate)
    up_convs = down_convs[0]
    for i in range(num_down_conv):
        filters = K.int_shape(down_convs[i+1])[-1]
        deconv = _createDeconv3D_(up_convs, filters, ksize, ds_stride, padding)
        if not resnet:
            up_convs = Lambda(_highwayBlock_)([deconv, down_convs[i+1], trans_gates[i+1]])
        else:
            up_convs = add([deconv, down_convs[i+1]])
    cost = _createDeconv3D_(up_convs, 1, ksize, ds_stride, padding, bn=False, act_func=None)
    cost = Lambda(lambda x: -x)(cost)
    cost = Lambda(K.squeeze, arguments={'axis': -1})(cost)
    return cost


def createFeature(input, layers):
    res = layers[0](input)
    tensor = res
    for layer in layers[1:]:
        if isinstance(layer, Add):
            tensor = layer([tensor, res])
            res = tensor
        else:
            tensor = layer(tensor)
    return tensor


def createGCNetwork(hp, tp, pre_weight):
    padding = 'same'
    cost_weight = tp['cost_volume_weight_path']
    linear_weight = tp['linear_output_weight_path']
    d = hp['max_disp']
    resnet = hp['resnet']
    first_ksize = hp['first_kernel_size']
    ksize = hp['kernel_size']
    num_filters = hp['base_num_filters']
    act_func = hp['act_func']
    highway_func = hp['h_act_func']
    num_down_conv = hp['num_down_conv']
    output = hp['output']
    num_res = hp['num_res']
    ds_stride = hp['ds_stride']
    padding = hp['padding']
    shared_weight = tp['shared_weight']
    K.set_image_data_format(hp['data_format'])
    input_shape = (None, None, 3)
    left_img = Input(input_shape, dtype="float32")
    right_img = Input(input_shape, dtype="float32")
    layers = _createUniFeature_(input_shape, num_res, num_filters, first_ksize, ksize, act_func, ds_stride, padding)
    l_feature = createFeature(left_img, layers)
    if shared_weight == 1:
        print "Use shared weight for first stage"
        r_feature = createFeature(right_img, layers)
    else:
        print "Use different weights for first stage"
        layers2 = _createUniFeature_(input_shape, num_res, num_filters, first_ksize, ksize, act_func, ds_stride, padding)
        r_feature = createFeature(right_img, layers2)
    unifeatures = [l_feature, r_feature]
    cv = Lambda(_getCostVolume_, arguments={'max_d': d/2}, output_shape=(d/2, None, None, num_filters * 2))(unifeatures)
    disp_map = _LearnReg_(cv, num_filters, ksize, ds_stride, resnet, padding, highway_func, num_down_conv)
    cost_model = Model([left_img, right_img], disp_map)
    if pre_weight == 1:
        print "Loading pretrained cost weight..."
        cost_model.load_weights(cost_weight)
    out_func = getOutputFunction(output)
    disp_map_input = Input((d, None, None))
    output = Lambda(out_func, arguments={'d': d})(disp_map_input)
    linear_output_model = Model(disp_map_input, output)
    if out_func == "linear" and pre_weight == 1:
        print "Loading pretrained linear output weight..."
        linear_output_model.load_weights(linear_weight)
    model = Model(cost_model.input, linear_output_model(cost_model.output))
    return model

