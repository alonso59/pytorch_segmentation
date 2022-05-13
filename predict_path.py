import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from patchify import patchify
import settings as cfg
from torchsummary import summary
import torch.nn.functional as F
import os
import multiprocessing as mp

def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0


def patching(img, size, shiftx, shifty):
    for i in range(size, img.shape[0], shifty):
        for j in range(size, img.shape[1], shiftx):
           pass


def plot_inference():
    files_images = get_filenames(cfg.VAL_IMAGES, 'tiff')
    files_masks = get_filenames(cfg.VAL_MASKS, 'tiff')

    np.random.seed(30)  # 2, 10, 20, 30 seed randome
    name = 'unet'
    x = np.random.randint(0, len(files_images))
    pick_image = files_images[x]

    large_image = np.array(Image.open(pick_image))
    patches_images = patchify(np.array(large_image), (128, 128), step=128)

    pick_mask = files_masks[x]
    large_mask = np.array(Image.open(pick_mask).convert('L'))
    patches_mask = patchify(np.array(large_mask), (128, 128), step=128)

    pred = np.zeros(patches_images.shape)

    for i in range(patches_images.shape[0]):
        for j in range(patches_images.shape[1]):
            single_image = patches_images[i, j, :, :]
            pred[i, j, :, :] = prediction(single_image, name)
    img1 = patches_images[0, :, :, :]
    img2 = patches_images[1, :, :, :]
    imgh1 = np.hstack((img1))
    imgh2 = np.hstack((img2))
    img = np.vstack((imgh1, imgh2))
    msk1 = patches_mask[0, :, :, :]
    msk2 = patches_mask[1, :, :, :]
    mskh1 = np.hstack((msk1))
    mskh2 = np.hstack((msk2))
    msk = np.vstack((mskh1, mskh2))
    pred1 = pred[0, :, :, :]
    pred2 = pred[1, :, :, :]
    predh1 = np.hstack((pred1))
    predh2 = np.hstack((pred2))
    predict = np.vstack((predh1, predh2))
    figure, ax = plt.subplots(nrows=3, ncols=1)
    figure.set_figwidth(12)
    figure.set_figheight(12)
    ax[0].imshow(img, cmap='gray')
    ax[0].title.set_text('Test image')
    ax[1].imshow(msk, cmap='jet')
    ax[1].title.set_text('Ground Truth')
    ax[2].imshow(predict, cmap='jet')
    ax[2].title.set_text('Prediction')
    figure.suptitle(name, fontsize=24)
    plt.savefig('dataset/' + os.path.splitext(os.path.split(pick_image)[1])[0] + '_' + name + '.tiff')
    # plt.show()


def inference_path(image):
    name = 'unet'
    large_image = np.array(Image.open(image))
    patches_images = patchify(np.array(large_image), (128, 128), step=124)
    pred = np.zeros((patches_images.shape[0], patches_images.shape[1], patches_images.shape[2], patches_images.shape[3], 3))
    for i in range(patches_images.shape[0]):
        for j in range(patches_images.shape[1]):
            single_image = patches_images[i, j, :, :]
            pred[i, j, :, :, :] = prediction(single_image, name)
    img1 = patches_images[0, :, :, :]
    img2 = patches_images[1, :, :, :]
    img3 = patches_images[3, :, :, :]
    imgh1 = np.hstack((img1))
    imgh2 = np.hstack((img2))
    imgh3 = np.hstack((img3))
    img = np.vstack((imgh1, imgh2, imgh3))
    img = Image.fromarray(img)
    img.save('dataset/preds/Images/' + os.path.splitext(os.path.split(image)[1])[0] + '_' +name + '.png')
    pred1 = pred[0, :, :, :]
    pred2 = pred[1, :, :, :]
    pred3 = pred[3, :, :, :]
    predh1 = np.hstack((pred1))
    predh2 = np.hstack((pred2))
    predh3 = np.hstack((pred3))
    pred = np.vstack((predh1, predh2, predh3)) * 255
    predict = pred.astype('uint8')
    predict = Image.fromarray(predict).convert('RGB')
    predict.save('dataset/preds/Masks/' + os.path.splitext(os.path.split(image)[1])[0] + '_' + name + '.png')


def prediction(image, name):
    """ CUDA device """
    device = torch.device("cuda")
    PATH = 'logs/version0/checkpoints/model.pth'
    image1 = np.array(Image.fromarray(image)) / 255.
    image1 = np.expand_dims(image1, axis=0)
    image1 = np.expand_dims(image1, axis=0)
    image1 = torch.tensor(image1, dtype=torch.float, device=device)
    best_model = torch.load(PATH)
    pr_mask = best_model(image1)
    pr_mask = F.softmax(pr_mask, dim=1)
    pr_mask = torch.argmax(pr_mask, dim=1)
    
    img_rgb = torch.zeros((3, pr_mask.size(1), pr_mask.size(2)), dtype=torch.float, device=device)
    
    img_rgb[0, :, :] = torch.where(pr_mask.squeeze(0) == 1, 1, 0)
    img_rgb[1, :, :] = torch.where(pr_mask.squeeze(0) == 2, 1, 0)
    img_rgb[2, :, :] = torch.where(pr_mask.squeeze(0) == 3, 1, 0)
    single_mask = (img_rgb.squeeze().cpu().long().detach().numpy())
    return single_mask.transpose(1, 2, 0)


def main():
    files = get_filenames(cfg.VAL_IMAGES, 'tiff')
    # plot_inference()
    for file in files:
        # print(file)
        inference_path(file)




if __name__ == '__main__':
    # plot_inference()
    main()