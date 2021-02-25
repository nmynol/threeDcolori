import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

Tran_3 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
Tran_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])


def not_first_frame(name):
    return int(name.split('.')[0].split('_')[-1]) != 1


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_img(path, img_size):
    # print(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # print(img.shape)
    # img = img.crop((0, img.size[0] * 7 // 32, img.size[0], img.size[0] * 25 // 32))
    img = cv2.resize(img, (img_size, img_size))
    # Tran = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # img = Tran(img)
    return img


def get_xdog(path, img_size):
    # print(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    # Tran = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(0.5, 0.5)
    # ])
    # img = Tran(img)
    return img


def get_diffs(imgs_now, img_first, xdogs_now, xdog_first, threshold):

    img_diffs = np.concatenate(
        [
            cv2.threshold(
                cv2.cvtColor(
                    cv2.absdiff(img_first, imgs_now[i])
                    , cv2.COLOR_RGB2GRAY)
                , threshold, 255, cv2.THRESH_BINARY)[1][np.newaxis, :]
            for i in range(imgs_now.shape[0])
        ]
        , 0)

    xdog_diffs = np.concatenate(
        [
            cv2.absdiff(xdog_first, xdogs_now[i])[np.newaxis, :]
            for i in range(imgs_now.shape[0])
        ]
        , 0)

    return img_diffs, xdog_diffs


def transform_pic(img_diffs, xdog_diffs, imgs_now, img_first, xdogs_now, xdog_first):

    img_diffs = torch.cat([Tran_1(img_diffs[i]).unsqueeze(0) for i in range(img_diffs.shape[0])], 0)
    xdog_diffs = torch.cat([Tran_1(xdog_diffs[i]).unsqueeze(0) for i in range(xdog_diffs.shape[0])], 0)
    imgs_now = torch.cat([Tran_1(imgs_now[i]).unsqueeze(0) for i in range(imgs_now.shape[0])], 0)
    xdogs_now = torch.cat([Tran_1(xdogs_now[i]).unsqueeze(0) for i in range(xdogs_now.shape[0])], 0)
    return img_diffs, xdog_diffs, imgs_now, Tran_3(img_first), xdogs_now, Tran_1(xdog_first)
