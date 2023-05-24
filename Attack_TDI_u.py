"""
Modulate the Adversarial perturbation for better undetectability.
Baseline: un-targeted TDI-FGSM
We use S-UNIWARD to generate the embedding suitability map.
For each original image, we craft the following four AEs:
TDIFGSM, W-TDIFGSM, A-TDIFGSM, WA-TDIFGSM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import Image
import imageio
import numpy as np
from utils import gkern, DI, load_ground_truth, Normalize
import scipy.io as scio
from skimage.io import imread
from matplotlib import pyplot as plt


channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

# 1 Load the source model #
# 1.1 Load a pretrained model
model_2 = models.resnet50(pretrained=True).eval()
# 1.2 Set it in the eval mode
for param in model_2.parameters():
    param.requires_grad = False
# 1.3 Using GPU
device = torch.device("cuda:0")
model_2.to(device)

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# 2 Load dataset #
# values are standard normalization for ImageNet images,
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])
image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

batch_size = 1
max_iterations = 50
input_path = './dataset/images/'
mat_path = './SuitabilityMap/S_UNIWARD/'       # Embedding suitability map
# only one sample is used in this demo
num_batches = int(np.ceil((len(image_id_list) - 999) / batch_size))
img_size = 299
epsilon = 16            # L_inf norm bound
lr = (epsilon/8) / 255  # step size
# White-box attack success numbers:
# row 0: TDI, row 1: Weighted TDI, row 2: Attentional TDI, row 3: WA TDI
pos = np.zeros((4, max_iterations // 50))

# 4 Crafting AE #
# 4.1 TDI with CE loss
for k in range(0, num_batches):
    print(k)
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
    labels_ori = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

    for t in range(max_iterations):
        # X_adv = X_ori + delta
        X_adv, _, _, _, _ = DI(X_ori + delta)                     # DI 2
        logits = model_2(norm(X_adv))
        # loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels_ori)  # un-targeted
        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI 3

        delta.grad.zero_()
        delta.data = delta.data + lr * torch.sign(grad_c)                # un-targeted
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
        if t % 50 == 49:
            X_adv = X_ori + delta
            X_adv_norm = norm(X_adv).detach()
            output2 = model_2(X_adv_norm)
            predict = torch.argmax(output2, dim=1)
            pos[0, t // 50] = pos[0, t // 50] + sum(predict != labels_ori).cpu().numpy()
            X_adv_cpu = X_adv.detach().cpu()
            for img_i in range(batch_size_cur):
                X_adv_img = X_adv_cpu[img_i].permute(1, 2, 0)
                save_path = 'advimgs_un/TDI/' + image_id_list[k * batch_size + img_i] + '.png'
                imageio.imwrite(save_path, X_adv_img)

Line = 1
torch.cuda.empty_cache()
# 4.2 CE + Weighted (S-UNIWARD)
for k in range(0, num_batches):
    print(k)
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    rho_ts = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
        saved_data = scio.loadmat(mat_path + image_id_list[k * batch_size + i] + '.mat')
        rho = saved_data['rho_UNIWARD']
        rho[rho > 10] = 10
        rho[rho < 1] = 1
        rho = (1. / rho)  # Xi
        rho = rho / rho.max()  # NXi
        rho_ts[i] = torch.from_numpy(np.stack([rho, rho, rho], axis=0)).unsqueeze(dim=0).cuda()
    labels_ori = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

    for t in range(max_iterations):
        # X_adv = X_ori + delta
        X_adv, _, _, _, _ = DI(X_ori + delta)  # DI 2
        logits = model_2(norm(X_adv))
        # loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels_ori)  # un-targeted
        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI 3

        delta.grad.zero_()
        weighted_grad = torch.sign(grad_c) * rho_ts     #
        delta.data = delta.data + lr * weighted_grad    # un-targeted
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
        if t % 50 == 49:
            X_adv = X_ori + delta
            X_adv_norm = norm(X_adv).detach()
            output2 = model_2(X_adv_norm)
            predict = torch.argmax(output2, dim=1)
            pos[1, t // 50] = pos[1, t // 50] + sum(predict != labels_ori).cpu().numpy()

            X_adv_cpu = X_adv.detach().cpu()
            for img_i in range(batch_size_cur):
                X_adv_img = X_adv_cpu[img_i].permute(1, 2, 0)
                save_path = 'advimgs_un/TDI_UNIWARD/' + image_id_list[k * batch_size + img_i] + '.png'  # 6
                imageio.imwrite(save_path, X_adv_img)

Line = 2
torch.cuda.empty_cache()
# 4.3 CE + CAM
for k in range(0, num_batches):
    print(k)
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    cam = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
        cam_name = 'CAM/'+image_id_list[k * batch_size + i] + '.png'
        cam_matrix_o = imread(cam_name)
        cam_matrix = cam_matrix_o.astype(np.float32) * 1.0
        bcam = cam_matrix > (255. / 3)
        cam[i] = torch.from_numpy(np.stack([bcam, bcam, bcam], axis=0)).unsqueeze(dim=0).cuda()
    labels_ori = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

    for t in range(max_iterations):
        # X_adv = X_ori + delta * cam
        X_adv, _, _, _, _ = DI(X_ori + delta * cam)  # DI 2
        logits = model_2(norm(X_adv))
        # loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels_ori)  # un-targeted
        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI 3

        delta.grad.zero_()
        cam_grad = torch.sign(grad_c) * cam
        delta.data = delta.data + lr * cam_grad
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
        if t % 50 == 49:
            X_adv = X_ori + delta
            X_adv_norm = norm(X_adv).detach()
            output2 = model_2(X_adv_norm)
            predict = torch.argmax(output2, dim=1)
            pos[2, t // 50] = pos[2, t // 50] + sum(predict != labels_ori).cpu().numpy()
            X_adv_cpu = X_adv.detach().cpu()
            for img_i in range(batch_size_cur):
                X_adv_img = X_adv_cpu[img_i].permute(1, 2, 0)
                save_path = 'advimgs_un/TDI_CAM/' + image_id_list[k * batch_size + img_i] + '.png'  # 6
                imageio.imwrite(save_path, X_adv_img)

Line = 3
torch.cuda.empty_cache()
# 4.4 CE + Weighted(S-UNIWARD) + CAM
for k in range(0, num_batches):
    print(k)
    batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    rho_cam_ts = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        img_ori = Image.open(input_path + image_id_list[k * batch_size + i] + '.png')
        X_ori[i] = trn(img_ori)
        saved_data = scio.loadmat(mat_path + image_id_list[k * batch_size + i] + '.mat')
        rho = saved_data['rho_UNIWARD']
        rho[rho > 10] = 10
        rho[rho < 1] = 1
        rho = (1. / rho)                        # Xi
        cam_name = 'CAM/'+image_id_list[k * batch_size + i] + '.png'
        cam_matrix_o = imread(cam_name)
        cam_matrix = cam_matrix_o.astype(np.float32) * 1.0
        bcam = cam_matrix > (255. / 3)          # Binarized attentional map
        rho_cam = rho * bcam                    # XiA
        if rho_cam.max() == 0:
            print('all-zero cam in textural area')
        else:
            rho_cam = rho_cam / (rho_cam.max())    # Normalization to NXiA
        plt.imshow(img_ori)
        plt.show()
        plt.imshow(bcam, cmap='gray')
        plt.show()
        plt.imshow(rho, cmap='gray')
        plt.show()
        plt.imshow(rho_cam, cmap='gray')
        plt.show()

        rho_cam_ts[i] = torch.from_numpy(np.stack([rho_cam, rho_cam, rho_cam], axis=0)).unsqueeze(dim=0).cuda()
    labels_ori = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

    for t in range(max_iterations):
        # X_adv = X_ori + delta * cam
        X_adv, _, _, _, _ = DI(X_ori + delta * cam)  # DI 2
        logits = model_2(norm(X_adv))
        # loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)
        loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels_ori)  # un-targeted
        loss.backward()
        grad_c = delta.grad.clone()
        grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI 3

        delta.grad.zero_()
        rho_cam_grad = torch.sign(grad_c) * rho_cam_ts
        delta.data = delta.data + lr * rho_cam_grad    # un-targeted
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
        if t % 50 == 49:
            X_adv = X_ori + delta
            X_adv_norm = norm(X_adv).detach()
            output2 = model_2(X_adv_norm)
            predict = torch.argmax(output2, dim=1)
            pos[3, t // 50] = pos[3, t // 50] + sum(predict != labels_ori).cpu().numpy()
            X_adv_cpu = X_adv.detach().cpu()
            for img_i in range(batch_size_cur):
                X_adv_img = X_adv_cpu[img_i].permute(1, 2, 0)
                save_path = 'advimgs_un/TDI_UNIWARD_CAM/' + image_id_list[k * batch_size + img_i] + '.png'  # 6
                imageio.imwrite(save_path, X_adv_img)

print(pos)
done = 1
