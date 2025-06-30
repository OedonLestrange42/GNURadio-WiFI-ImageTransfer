import torch
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import PIL

from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from torchvision import transforms
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

random.seed(0)


def psnr(img1, img2):
    epsilon = 1e-4
    psnr = 0
    bs = img1.shape[0]
    if not bs == img2.shape[0]:
        Exception("img1 and img2 have different batch size")
    else:
        for b in range(bs):
            mse = np.mean((img1[b]-img2[b])**2)
            mse = np.max([epsilon, mse])
            psnr += 10.0 * np.log10(255.0 * 255.0 / mse)
    return psnr / bs


def denormalize(img):
    """

    :param img: tensor [b, c, w, h] or [c, w, h]
    :return: denormalized tensor
    """
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    img_norm = torch.zeros_like(img)
    if len(img.shape) == 4:
        img_norm[:, 0, :, :] = std[0] * img[:, 0, :, :] + mean[0]
        img_norm[:, 1, :, :] = std[1] * img[:, 1, :, :] + mean[1]
        img_norm[:, 2, :, :] = std[2] * img[:, 2, :, :] + mean[2]
    elif len(img.shape) == 3:
        img_norm[0] = std[0] * img[0] + mean[0]
        img_norm[1] = std[1] * img[1] + mean[1]
        img_norm[2] = std[2] * img[2] + mean[2]
    else:
        Exception("input must have 3 or 4 channel")

    return img_norm


def rgb2ycbcr(rgb_image: np.ndarray) -> np.ndarray:
    assert rgb_image.shape[-1] == 3, "RGB image does not have three channels in the last dimension"
    assert rgb_image.dtype == np.uint8, "RGB image doesn't have the correct data type"
    weights = np.zeros(shape=(3, 3), dtype=np.float32)
    weights[0] = (65.481 / 255.0, 128.553 / 255.0, 24.944 / 255.0)
    weights[1] = (-37.797 / 255.0, -74.203 / 255.0, 112.0 / 255.0)
    weights[2] = (112.0 / 255.0, -93.786 / 255.0, -18.214 / 255.0)
    bias = np.array((16.0, 128.0, 128.0), dtype=np.float32)
    return np.clip(np.matmul(rgb_image.astype(np.float32), weights.T) + bias, 16, 255).astype(np.uint8)


@torch.inference_mode()
def evaluate(net, dataloader, batch_size, user_num=2, metric='PSNR', channel_axis=2, device=torch.device('cpu'), save_result=False):
    toPIL = transforms.ToPILImage()
    net.eval()
    num_val_batches = len(dataloader)
    bs_per_user = batch_size
    average_psnr = 0

    # iterate over the validation set
    for image, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', position=0, leave=True):
        image = image.to(device)

        output = net(image)

        restored = np.clip(denormalize(output).detach().cpu().numpy(), 0, 1) * 255
        source = np.clip(denormalize(image).detach().cpu().numpy(), 0, 1) * 255
        if metric == 'PSNR':
            PSNR = psnr(restored.astype(np.uint8), source.astype(np.uint8))
        else:
            if len(restored.shape) > 3:
                mtrc = 0
                for bth in range(restored.shape[0]):
                    mtrc += ssim(restored[bth].astype(np.uint8), source[bth].astype(np.uint8), channel_axis=channel_axis)
                PSNR = mtrc / restored.shape[0]
            else:
                PSNR = ssim(restored.astype(np.uint8), source.astype(np.uint8), channel_axis=channel_axis)
        average_psnr += PSNR

    # visualize
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_index = random.randint(0, batch_size - 1)
    img_ts = []
    for i in range(user_num):
        img_ts.extend([image[save_index + bs_per_user * i, :, :, :],
                       output[save_index + bs_per_user * i, :, :, :]])

    img_np = [np.array(toPIL(torch.clamp(I, 0, 1))) for I in img_ts]
    # img_np = [np.array(toPIL(I)) for I in img_ts]

    rows = user_num
    cols = 2
    Inch = img_np[0].shape[0] // 16  # why 16? empirical param for img display
    fig = plt.figure(figsize=(cols * Inch, rows * Inch))
    for i in range(1, rows * cols + 1):
        img_array = img_np[i - 1]
        # 子图位置
        ax = fig.add_subplot(rows, cols, i)
        ax.axis('off')  # 去掉每个子图的坐标轴
        ax.imshow(img_array)

    plt.subplots_adjust(wspace=0, hspace=0.1)  # 修改子图之间的间隔
    plt.tight_layout()

    dir_valResults = './validation_results/'
    Path(dir_valResults).mkdir(parents=True, exist_ok=True)
    if save_result:
        plt.savefig(dir_valResults + time_stamp + '.jpg', dpi=400)

    average_psnr = average_psnr / max(num_val_batches, 1)
    if metric == 'PSNR':
        print("#" + time_stamp + "#: PSNR = " + str(average_psnr))
    else:
        print("#" + time_stamp + "#: SSIM = " + str(average_psnr))
    net.train()

    # matplotlib 2 PIL
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # 获取绘制的图像数据
    buffer = canvas.buffer_rgba()
    width, height = canvas.get_width_height()
    # 创建PIL图像对象
    Img = PIL.Image.frombytes('RGBA', (width, height), buffer.tobytes())

    plt.close(fig)
    return average_psnr, Img


@torch.inference_mode()
def evaluate_M2M(net, dataloader, batch_size, user_num=2, metric='PSNR', channel_axis=2, device=torch.device('cpu'), save_result=False):
    toPIL = transforms.ToPILImage()
    net.eval()
    num_val_batches = len(dataloader)
    bs_per_user = batch_size
    average_psnr = 0

    # iterate over the validation set
    for image, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', position=0, leave=True):
        image = image.to(device)

        output = net(image)

        restored = [np.clip(denormalize(o).detach().cpu().numpy(), 0, 1) * 255 for o in output]
        source = np.clip(denormalize(image).detach().cpu().numpy(), 0, 1) * 255
        PSNR_tx = 0
        for u in range(user_num):
            if metric == 'PSNR':
                PSNR = psnr(restored[u].astype(np.uint8), source.astype(np.uint8))
            else:
                if len(restored.shape) > 3:
                    mtrc = 0
                    for bth in range(restored.shape[0]):
                        mtrc += ssim(restored[bth].astype(np.uint8), source[bth].astype(np.uint8), channel_axis=channel_axis)
                    PSNR = mtrc / restored.shape[0]
                else:
                    PSNR = ssim(restored.astype(np.uint8), source.astype(np.uint8), channel_axis=channel_axis)
            PSNR_tx += PSNR
        average_psnr += PSNR_tx / user_num

    # visualize
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_index = random.randint(0, batch_size - 1)
    img_ts = []
    for i in range(user_num-1):
        tmp = [output[u][save_index + bs_per_user * i, :, :, :] for u in range(user_num)]
        tmp.insert(0, image[save_index + bs_per_user * i, :, :, :])
        img_ts.extend(tmp)

    img_np = [np.array(toPIL(torch.clamp(I, 0, 1))) for I in img_ts]
    # img_np = [np.array(toPIL(I)) for I in img_ts]

    rows = user_num - 1
    cols = user_num + 1
    Inch = img_np[0].shape[0] // 16  # why 16? empirical param for img display
    fig = plt.figure(figsize=(cols * Inch, rows * Inch))
    for i in range(0, rows * cols):
        img_array = img_np[i]
        # 子图位置
        ax = fig.add_subplot(rows, cols, i+1)
        ax.axis('off')  # 去掉每个子图的坐标轴
        ax.imshow(img_array)

    plt.subplots_adjust(wspace=0, hspace=0.1)  # 修改子图之间的间隔
    plt.tight_layout()

    dir_valResults = './validation_results/'
    Path(dir_valResults).mkdir(parents=True, exist_ok=True)
    if save_result:
        plt.savefig(dir_valResults + time_stamp + '.jpg', dpi=400)

    average_psnr = average_psnr / max(num_val_batches, 1)
    if metric == 'PSNR':
        print("#" + time_stamp + "#: PSNR = " + str(average_psnr))
    else:
        print("#" + time_stamp + "#: SSIM = " + str(average_psnr))
    net.train()

    # matplotlib 2 PIL
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # 获取绘制的图像数据
    buffer = canvas.buffer_rgba()
    width, height = canvas.get_width_height()
    # 创建PIL图像对象
    Img = PIL.Image.frombytes('RGBA', (width, height), buffer.tobytes())

    plt.close(fig)
    return average_psnr, Img


@torch.inference_mode()
def evaluate_SOMA(net, dataloader, batch_size, pair_list, metric='PSNR', channel_axis=1, device=torch.device('cpu'), save_result=False):
    toPIL = transforms.ToPILImage()
    net.eval()
    num_val_batches = len(dataloader)
    average_psnr = 0

    # iterate over the validation set
    for image, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', position=0, leave=True):
        image = image.to(device)
        image4each = torch.split(image, batch_size, dim=0)
        schedule_list = {}
        for pair_index in range(len(pair_list)):
            schedule_list[pair_list[pair_index]] = image4each[pair_index]

        output = net(schedule_list)
        if isinstance(output, tuple):
            output = output[0]
        PSNR = 0

        if hasattr(net, 'semantic_model'):
            for key in output.keys():
                output[key] = output[key][0]

        for i, (key, reconstruct) in enumerate(output.items()):
            rebuild = np.clip(denormalize(reconstruct).detach().cpu().numpy(), 0, 1) * 255
            source = np.clip(denormalize(schedule_list[key]).detach().cpu().numpy(), 0, 1) * 255

            if metric == 'PSNR':
                PSNR += psnr(rebuild.astype(np.uint8), source.astype(np.uint8))
            else:
                PSNR += ssim(rebuild.astype(np.uint8), source.astype(np.uint8), channel_axis=channel_axis)
        average_psnr += PSNR / len(schedule_list)

    # visualize
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_index = random.randint(0, batch_size - 1)
    # 计算需要的子图数量
    num_images = len(output)
    num_cols = 4  # 每行最多显示 4 张图片
    num_rows = np.ceil(num_images / num_cols).astype(int)

    # 创建figure和axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = np.atleast_2d(axes)

    # 遍历字典,在子图中显示图片并添加标注
    for i, (key, image) in enumerate(output.items()):
        row, col = i // num_cols, i % num_cols
        ax = axes[row, col]
        pic = np.array(toPIL(torch.clamp(denormalize(image[save_index]), 0, 1)))
        ax.imshow(pic)
        ax.set_title(key)
        ax.axis('off')

    # 调整布局并显示图片
    plt.subplots_adjust(wspace=0, hspace=0.1)  # 修改子图之间的间隔
    plt.tight_layout()

    dir_valResults = './validation_results/'
    Path(dir_valResults).mkdir(parents=True, exist_ok=True)
    if save_result:
        plt.savefig(dir_valResults + time_stamp + '.jpg', dpi=400)

    average_psnr = average_psnr / max(num_val_batches, 1)
    if metric == 'PSNR':
        print("#" + time_stamp + "#: PSNR = " + str(average_psnr))
    else:
        print("#" + time_stamp + "#: SSIM = " + str(average_psnr))
    net.train()

    # matplotlib 2 PIL
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # 获取绘制的图像数据
    buffer = canvas.buffer_rgba()
    width, height = canvas.get_width_height()
    # 创建PIL图像对象
    Img = PIL.Image.frombytes('RGBA', (width, height), buffer.tobytes())

    plt.close(fig)
    return average_psnr, Img


def evaluate_MultiBand(net, dataloader, batch_size, pair_list_multiband, metric='PSNR', channel_axis=1, device=torch.device('cpu'), save_result=False):
    toPIL = transforms.ToPILImage()
    net.eval()
    num_val_batches = len(dataloader)
    average_psnr = 0

    # iterate over the validation set
    output = {}
    sources = {}
    for image, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', position=0, leave=True):
        image = image.to(device)
        image4each = torch.split(image, batch_size, dim=0)

        schedule_list = {}
        for _, freq in enumerate(pair_list_multiband):
            schedule_list[freq] = {}
            for pair in pair_list_multiband[freq]:
                schedule_list[freq][pair] = image4each[int(pair.split('-')[0])]

        net_output = net(schedule_list)
        for _, f in enumerate(net_output):
            for _, o in enumerate(net_output[f]):
                key = str(f) + '-' + o
                output[key] = net_output[f][o]
                sources[key] = schedule_list[f][o]
        PSNR = 0

        if hasattr(net, 'semantic_model'):
            for key in output.keys():
                output[key] = output[key][0]

        for i, (key, reconstruct) in enumerate(output.items()):
            rebuild = np.clip(denormalize(reconstruct).detach().cpu().numpy(), 0, 1) * 255
            source = np.clip(denormalize(sources[key]).detach().cpu().numpy(), 0, 1) * 255

            if metric == 'PSNR':
                PSNR += psnr(rebuild.astype(np.uint8), source.astype(np.uint8))
            else:
                PSNR += ssim(rebuild.astype(np.uint8), source.astype(np.uint8), channel_axis=channel_axis)
        average_psnr += PSNR / len(output)

    # visualize
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_index = random.randint(0, batch_size - 1)
    # 计算需要的子图数量
    num_images = len(output)
    num_cols = 4  # 每行最多显示 4 张图片
    num_rows = np.ceil(num_images / num_cols).astype(int)

    # 创建figure和axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = np.atleast_2d(axes)

    # 遍历字典,在子图中显示图片并添加标注
    for i, (key, image) in enumerate(output.items()):
        row, col = i // num_cols, i % num_cols
        ax = axes[row, col]
        pic = np.array(toPIL(torch.clamp(denormalize(image[save_index]), 0, 1)))
        ax.imshow(pic)
        ax.set_title(key)
        ax.axis('off')

    # 调整布局并显示图片
    plt.subplots_adjust(wspace=0, hspace=0.1)  # 修改子图之间的间隔
    plt.tight_layout()

    dir_valResults = './validation_results/'
    Path(dir_valResults).mkdir(parents=True, exist_ok=True)
    if save_result:
        plt.savefig(dir_valResults + time_stamp + '.jpg', dpi=400)

    average_psnr = average_psnr / max(num_val_batches, 1)
    if metric == 'PSNR':
        print("#" + time_stamp + "#: PSNR = " + str(average_psnr))
    else:
        print("#" + time_stamp + "#: SSIM = " + str(average_psnr))
    net.train()

    # matplotlib 2 PIL
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # 获取绘制的图像数据
    buffer = canvas.buffer_rgba()
    width, height = canvas.get_width_height()
    # 创建PIL图像对象
    Img = PIL.Image.frombytes('RGBA', (width, height), buffer.tobytes())

    plt.close(fig)
    return average_psnr, Img


def test_MultiBand(net, dataloader, batch_size, pair_list_multiband, metric='PSNR', channel_axis=1, device=torch.device('cpu'), save_result=False):
    toPIL = transforms.ToPILImage()
    net.eval()
    num_val_batches = len(dataloader)
    average_psnr = 0

    # iterate over the validation set
    output = {}
    sources = {}
    for image, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', position=0, leave=True):
        image = image.to(device)
        image4each = torch.split(image, batch_size, dim=0)

        schedule_list = {}
        img_index = 0
        for _, freq in enumerate(pair_list_multiband):
            schedule_list[freq] = {}
            for pair in pair_list_multiband[freq]:
                schedule_list[freq][pair] = image4each[img_index]
                img_index += 1

        net_output = net(schedule_list)

        for _, f in enumerate(net_output):
            for _, o in enumerate(net_output[f]):
                key = str(f) + '-' + o
                output[key] = net_output[f][o]
                sources[key] = schedule_list[f][o]


        if hasattr(net, 'semantic_model'):
            for key in output.keys():
                output[key] = output[key][0]

        output_ssim = {}
        PSNR = 0
        for i, (key, reconstruct) in enumerate(output.items()):
            rebuild = np.clip(denormalize(reconstruct).detach().cpu().numpy(), 0, 1) * 255
            source = np.clip(denormalize(sources[key]).detach().cpu().numpy(), 0, 1) * 255

            if metric == 'PSNR':
                loss = psnr(rebuild.astype(np.uint8), source.astype(np.uint8))
            else:
                loss = ssim(rebuild.astype(np.uint8), source.astype(np.uint8), channel_axis=channel_axis)
            output_ssim[key] = f"{metric}={loss:.2f}"
            PSNR += loss
        average_psnr += PSNR / len(output)

    # visualize
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_index = random.randint(0, batch_size - 1)
    # 计算需要的子图数量
    num_images = len(output)
    num_cols = 4  # 每行最多显示 4 张图片
    num_rows = np.ceil(num_images / num_cols).astype(int)

    # 创建figure和axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = np.atleast_2d(axes)

    # 遍历字典,在子图中显示图片并添加标注
    for i, (key, image) in enumerate(output.items()):
        row, col = i // num_cols, i % num_cols
        ax = axes[row, col]
        pic = np.array(toPIL(torch.clamp(denormalize(image[save_index]), 0, 1)))
        ax.imshow(pic)
        ax.set_title(f"{key}-{output_ssim[key]}")
        ax.axis('off')

    # 调整布局并显示图片
    plt.subplots_adjust(wspace=0, hspace=0.1)  # 修改子图之间的间隔
    plt.tight_layout()

    dir_valResults = './validation_results/'
    Path(dir_valResults).mkdir(parents=True, exist_ok=True)
    if save_result:
        plt.savefig(dir_valResults + time_stamp + '.jpg', dpi=400)

    average_psnr = average_psnr / max(num_val_batches, 1)
    if metric == 'PSNR':
        print("#" + time_stamp + "#: PSNR = " + str(average_psnr))
    else:
        print("#" + time_stamp + "#: SSIM = " + str(average_psnr))
    net.train()

    # matplotlib 2 PIL
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # 获取绘制的图像数据
    buffer = canvas.buffer_rgba()
    width, height = canvas.get_width_height()
    # 创建PIL图像对象
    Img = PIL.Image.frombytes('RGBA', (width, height), buffer.tobytes())

    plt.close(fig)
    return average_psnr, Img

