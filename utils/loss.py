import torch
from utils.ms_ssim import *
import math
import torch.nn.functional
from utils.vgg import Vgg16

l2_loss_fn = torch.nn.MSELoss(reduction='mean').cuda()
losser = MS_SSIM(max_val=1, channel=3).cuda()
t_losser = MS_SSIM(max_val=1, channel=1).cuda()
loss_mse = torch.nn.MSELoss()
vgg = Vgg16().type(torch.cuda.FloatTensor).cuda()


def l2_loss(output, gth):
    return l2_loss_fn(output, gth)


def ssim_loss(output, gth, channel=3):
    # losser = MS_SSIM(data_range=1.).cuda()
    return 1 - losser(output, gth)


def vgg_loss(output, gth):
    # vgg = Vgg16().cuda()
    output_features = vgg(output)
    gth_features = vgg(gth)
    sum_loss = loss_mse(output_features[0], gth_features[0]) * 0.25 \
               + loss_mse(output_features[1], gth_features[1]) * 0.25 \
               + loss_mse(output_features[2], gth_features[2]) * 0.25 \
               + loss_mse(output_features[3], gth_features[3]) * 0.25
    return sum_loss


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
    # J, A, t, gt_image, A_gth, t_gth, J_reconstruct, haze_reconstruct, haze_image
    J, A, t, gt_image, A_gth, t_gth, J_reconstruct, haze_reconstruct, haze_image = image
    # print(A.size(), A_gth.size())
    loss_train = [l2_loss(A, A_gth),
                  l2_loss(t, t_gth), 1 - t_losser(t, t_gth),
                  l2_loss(J, gt_image),
                  ssim_loss(J, gt_image),
                  vgg_loss(J, gt_image),
                  l2_loss(J_reconstruct, gt_image),
                  ssim_loss(J_reconstruct, gt_image),
                  vgg_loss(J_reconstruct, gt_image),
                  l2_loss(haze_reconstruct, haze_image),
                  ssim_loss(haze_reconstruct, haze_image),
                  vgg_loss(haze_reconstruct, haze_image)]
    # vgg_loss(J, gt_image)
    loss_sum = 0
    for i in range(len(loss_train)):
        loss_sum = loss_sum + loss_train[i] * weight[i]
        loss_train[i] = loss_train[i].item()
        # print("i=%d" % i)
        # print(loss_train[i])
        # print(loss_train[i] * weight[i])
    return loss_sum, loss_train
