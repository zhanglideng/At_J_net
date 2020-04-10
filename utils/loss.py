import torch
from utils.ms_ssim import *
import math
import torch.nn.functional
from utils.vgg import Vgg16


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
    output_features_1, output_features_2, output_features_3, output_features_4 = vgg(output)
    gth_features_1, gth_features_2, gth_features_3, gth_features_4 = vgg(gth)
    sum_loss = l2_loss(output_features_1, gth_features_1) + \
               l2_loss(output_features_2, gth_features_2) + \
               l2_loss(output_features_3, gth_features_3) + \
               l2_loss(output_features_4, gth_features_4)
    return sum_loss / 4


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
    J, gt_image = image
    loss_train = [l2_loss(J, gt_image),
                  ssim_loss(J, gt_image),
                  vgg_loss(J, gt_image)]
    # vgg_loss(J, gt_image)
    loss_sum = 0
    for i in range(len(loss_train)):
        loss_sum = loss_sum + loss_train[i] * weight[i]
    return loss_sum, loss_train
