import torch
from utils.ms_ssim import *
import math
import torch.nn.functional
from vgg import Vgg16


def l2_loss(output, gth):
    l2_loss_fn = torch.nn.MSELoss(reduction='mean').cuda()
    return l2_loss_fn(output, gth) * 100


def ssim_loss(output, gth, channel=3):
    losser = MS_SSIM(max_val=1, channel=channel).cuda()
    # losser = MS_SSIM(data_range=1.).cuda()
    return (1 - losser(output, gth)) * 100


def vgg_loss(output, gth):
    vgg = Vgg16().type(torch.cuda.FloatTensor)
    # vgg = Vgg16().cuda()
    output_features = vgg(output)
    gth_features = vgg(gth)
    return l2_loss(output_features, gth_features)


'''
def color_loss(image, label, len_reg=0):
    vec1 = tf.reshape(image, [-1, 3]) 想要得到一个第二维度是3的向量，不管第一维度是几。
    vec2 = tf.reshape(label, [-1, 3])
    clip_value = 0.999999
    norm_vec1 = tf.nn.l2_normalize(vec1, 1) l2范化
    norm_vec2 = tf.nn.l2_normalize(vec2, 1)
    dot = tf.reduce_sum(norm_vec1*norm_vec2, 1) 维度求和
    dot = tf.clip_by_value(dot, -clip_value, clip_value) 限制在一个范围内
    angle = tf.acos(dot) * (180/math.pi) 根据值反求角度
    return tf.reduce_mean(angle) 计算角度的均值
'''


def color_loss(input_image, output_image):
    vec1 = input_image.view([-1, 3])
    vec2 = output_image.view([-1, 3])
    clip_value = 0.999999
    norm_vec1 = torch.nn.functional.normalize(vec1)
    norm_vec2 = torch.nn.functional.normalize(vec2)
    dot = norm_vec1 * norm_vec2
    dot = dot.mean(dim=1)
    dot = torch.clamp(dot, -clip_value, clip_value)
    angle = torch.acos(dot) * (180 / math.pi)
    return angle.mean()


def loss_function(image, weight):
    gt_image, A_image, t_image, J, A, t = image
    loss_train = [l2_loss(gt_image, J),
                  ssim_loss(gt_image, J),
                  l2_loss(A_image, A),
                  l2_loss(t_image, t),
                  ssim_loss(t_image, t, channel=1)]
    loss_sum = 0
    for i in range(len(loss_train)):
        loss_sum = loss_sum + loss_train[i] * weight[i]
    return loss_sum, loss_train
