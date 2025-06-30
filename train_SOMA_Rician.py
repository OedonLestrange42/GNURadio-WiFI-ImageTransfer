import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import PIL
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
import pdb

from typing import Union
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from utils.SV_channel import Saleh_Valenzuela_Channel
from utils.dataloader import ImagenetMini
from utils.validation import evaluate_SOMA
from utils.Trainer import Trainer
from pathlib import Path
from datetime import datetime

np.random.seed(0)
dir_checkpoint = Path('codec/checkpoints/')


def upgrade_model(target, new):
    """

    :param target: model that needed to be upgrade
    :param new: new model weight
    :return: None
    """
    try:
        for name, param in new.state_dict().items():
            target.state_dict()[name].copy_(param)
        target.envs = new.envs
        # print(' # model upgraded! # ')
    except Exception:
        print("Error occurred when upgrading model")
        pdb.set_trace()


def print_CSI(csi):
    symbol_list = list(markers.MarkerStyle.markers.keys())

    fig, ax = plt.subplots()
    count = 0
    for c in csi:
        symbol = symbol_list[count % len(symbol_list)]
        count += 1
        ax.scatter(c[0], c[1], marker=symbol, color='black', label=f'User {count}')
    ax.legend()
    ax.set_xlabel('Real')
    ax.set_ylabel('Image')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title('CSI')

    # matplotlib 2 PIL
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # 获取绘制的图像数据
    buffer = canvas.buffer_rgba()
    width, height = canvas.get_width_height()
    # 创建PIL图像对象
    Img = PIL.Image.frombytes('RGBA', (width, height), buffer.tobytes())
    # plt.show()
    plt.close(fig)

    return Img


def generate_schedule_matrix(n, p=0.5):
    """
    生成一个 n x n 的特殊随机 01 矩阵,
    其中如果第 i 行有非 0 元素,则第 i 列全为 0,
    如果第 i 列有非 0 元素,则第 i 行全为 0.
    """
    matrix = np.ones((n, n), dtype=int)

    # 随机选择行或列,并将其全部设置为 1
    for i in range(n):
        if np.random.randint(2):  # 随机选择行或列
            matrix[i, :] = 0  # 将第 i 行全部设置为 1
        else:
            matrix[:, i] = 0  # 将第 i 列全部设置为 1
    matrix[np.random.rand(*matrix.shape) < p] = 0
    return matrix


def get_schedule_list(matrix):
    """
    获取给定矩阵中非零元素的位置坐标,以 "行-列" 的形式返回.
    """
    positions = []
    rows, cols = np.nonzero(matrix)
    for i, j in zip(rows, cols):
        positions.append(f"{i}-{j}")
    return positions


def train(
        expname,
        model,
        temp_model,
        trainer,
        chnl,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        fix_LR: bool = False,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        droprate: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        pretrained: str = "",
        schedule_type: Union[str, list] = "general",
        q: int = 0,
):
    # 0. If there is pretrained weight
    if len(pretrained) > 0:
        try:
            model.load_state_dict(torch.load(pretrained, map_location=device))
        except Exception:
            try:
                # init params
                checkpoint = torch.load(pretrained, map_location=device)
                shared_encoder_state_dict = {k[len('shared_encoder.'):]: v for k, v in checkpoint.items() if
                                             k.startswith('shared_encoder.')}
                shared_decoder_state_dict = {k[len('shared_decoder.'):]: v for k, v in checkpoint.items() if
                                             k.startswith('shared_decoder.')}

                model.shared_encoder.load_state_dict(shared_encoder_state_dict)
                model.shared_decoder.load_state_dict(shared_decoder_state_dict)
            except Exception:
                raise Exception(" # Error occurred when loading pretrained model # ")

    # 1. Create dataset
    img_size = int(256*img_scale)  # 64 for training from nothing
    transform = transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.Resize((img_size, img_size)),  # 调整图片大小以匹配AlexNet结构
        transforms.ToTensor(),  # 将图片转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    # 2. Split into train / validation partitions
    train_data = ImagenetMini(image_dir='/home/wanghd7/DataSets', transform=transform)
    test_data = ImagenetMini(image_dir='/home/wanghd7/DataSets', transform=transform)
    # cifar 10
    # train_data = datasets.CIFAR10(root="/home/wanghd7/DataSets", train=True, transform=transform, download=False)
    # test_data = datasets.CIFAR10(root="/home/wanghd7/DataSets", train=False, transform=transform, download=False)
    n_train = len(train_data)

    # visualize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.now().strftime("%Y%m%d")
    exp_name = expname
    run_name = "run-Rician_" + ("fix-LR_" if fix_LR else "") + exp_name + f"_{timestamp}"
    experiment = wandb.init(project='SemMA_SOMA-DSCN-exp-ver_1-bit', entity='oedon42', resume='allow', anonymous='must', name=run_name)
    if not trainer.dynamic_usr_num and not trainer.dynamic_usr_pos:
        experiment.config.update(
            dict(user_position=str(trainer.Usr_pos),
                 epochs=epochs,
                 batch_size=batch_size,
                 learning_rate=learning_rate,
                 save_checkpoint=save_checkpoint,
                 img_scale=img_scale,
                 amp=amp)
        )
    else:
        experiment.config.update(
            dict(epochs=epochs,
                 batch_size=batch_size,
                 learning_rate=learning_rate,
                 save_checkpoint=save_checkpoint,
                 img_scale=img_scale,
                 amp=amp)
        )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0
    hist_metric = -1000
    stagenate_count = 0
    for epoch in range(1, epochs + 1):
        SNR = list(np.random.randint(low=1, high=25, size=(user_number,)))
        if hasattr(model, 'encoder'):
            eps = random.random()
            if eps > droprate:
                # randomly set all users with identical SNR. This is for DeepMA only, enhancing the generality.
                SNR = [np.random.randint(low=1, high=25)] * user_number
        model.snr = SNR
        temp_model.snr = SNR

        # reset environment
        if trainer.dynamic_usr_num and trainer.dynamic_usr_pos:
            env, SNR = trainer.reset()
            model.reset(env, SNR)
            temp_model.reset(env, SNR)

        # create data loaders with dynamic batchsize according to usernum
        usrNum = len(model.envs)

        # setup optimizer
        upgrade_model(temp_model, model)
        optimizer = optim.Adam(temp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # schedule policy
        if isinstance(schedule_type, str):
            if schedule_type == 'general':
                pair_list = [str(x) + '-' + str(y) for i, x in enumerate(list(range(usrNum))) for j, y in
                             enumerate(list(range(usrNum))) if j > i]
            elif schedule_type == 'random':
                schedule_matrix = generate_schedule_matrix(usrNum)
                pair_list = get_schedule_list(schedule_matrix)
        else:
            pair_list = schedule_type

        link_num =  len(pair_list)
        if link_num == 0:
            continue

        loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size * link_num, **loader_args, drop_last=True)
        val_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size * link_num, **loader_args, drop_last=True)

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            # begin training
            for image, _ in train_loader:
                # random generate nLoS part in Rician channel
                new_envs = chnl.genU2R_Rician(10)
                new_envs = np.split(new_envs, user_number, axis=1)
                temp_model.refresh_rician_channel(new_envs)

                eps = random.random()
                if eps > droprate:
                    index = random.randint(0, batch_size * link_num-1)
                    tran = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    zero_pic = tran(torch.zeros_like(image[index]))
                    image[index] = zero_pic

                image = image.to(device)
                image4each = torch.split(image, batch_size, dim=0)

                # quantize IRS
                if not q == 0:
                    phi = temp_model.shared_phi.detach().cpu().numpy().flatten()
                    Psi = np.exp(1j * phi)
                    phi_new = np.angle(Psi)
                    phi_q = np.round(phi_new / (np.pi / q)) * (np.pi / q)
                    phi_q = torch.FloatTensor(phi_q).to(device)
                    temp_model.shared_phi = nn.Parameter(phi_q, requires_grad=True)

                schedule_list = {}
                for pair_index in range(link_num):
                    schedule_list[pair_list[pair_index]] = image4each[pair_index]

                loss = trainer.train(temp_model, schedule_list, mode='m2m')

                # BP
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(temp_model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # update wandb
                pbar.update(image.shape[0])
                global_step += 1
                experiment.log({
                    'train loss': loss,
                    'step': global_step,
                    'epoch': epoch
                })

        # data for visualization
        val_psnr, val_result = evaluate_SOMA(temp_model, val_loader, batch_size, pair_list, device=device)

        # update DNN parameters
        if fix_LR:
            upgrade_model(model, temp_model)
        else:
            if epoch < 100:
                upgrade_model(model, temp_model)
                hist_metric = val_psnr
            else:
                # if metric is higher, upgrade model
                if val_psnr > hist_metric:
                    upgrade_model(model, temp_model)
                    hist_metric = val_psnr
                    stagenate_count = 0
                else:
                    stagenate_count += 1
                    if stagenate_count >= 5:
                        learning_rate *= 0.5
                    stagenate_count = 0

        # visualize
        histograms = {}
        for tag, value in temp_model.named_parameters():
            if value.grad is not None:
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        if epoch % 10 == 0:
            try:
                experiment.log({
                    'images': wandb.Image(val_result),
                    'step': global_step,
                    'epoch': epoch,
                    'PSNR': val_psnr,
                    **histograms
                })
                # print(" - wandb log updated - ")
            except Exception as e:
                print(e)

        if save_checkpoint and epoch % 100 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / f'Rician-checkpoint_{exp_name}_epoch{epoch}_{date}.pth'))


def get_args():
    parser = argparse.ArgumentParser(description='Train the Encoder on images and target masks')
    parser.add_argument("--mode", '-m',
                        choices=["DeepMA",
                                 "SOMA",
                                 "SOMA-DSCN"],
                        default="SOMA-DSCN",
                        help="Specify the SemMA mode\n"
                             "DeepMA: existing work\n"
                             "SOMA: the proposed scheme\n"
                        )
    parser.add_argument('--noIRS', '-r', action='store_true', help='whether the IRS is taken into consideration')
    parser.add_argument('--optIRS', '-o', action='store_true', help='whether optimize IRS with BP')
    parser.add_argument('--IRS-scale', '-is', type=int, default=16, help='scale of IRS')
    parser.add_argument('--AP-antenna', '-a', metavar='AP-attenna-number', type=int, default=1, help='Number of AP antennas')
    parser.add_argument('--user-number', '-u', metavar='User-number', type=int, default=5, help='Number of users')

    parser.add_argument("--schedule", '-sch',
                        choices=["general", "random"],
                        default="general",
                        help="schedule type for training"
                        )
    parser.add_argument('--MinCSPP', '-mc', metavar='CSPP', type=float, default=None, help='Minimum CSPP')
    parser.add_argument('--img-scale', '-ps', type=float, default=.25, help='scale coefficient for 256*256 images')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=400, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=5e-5, help='Learning rate', dest='lr')
    parser.add_argument('--drop-rate', '-d', metavar='DR', type=float, default=0.5, help='Drop rate for img transmission', dest='dr')
    parser.add_argument('--fix-lr', '-fx', metavar='fixLR', type=bool, default=False, help='Wether fix learning rate', dest='fix_lr')
    parser.add_argument('--load', '-f', type=str,
                        default="",
                        help='Load model from a .pth file')
    # codec/checkpoints/Rician-checkpoint_SOMA-DSCN_withIRS_optIRS_IRS-scale-8_AP-1_Usr-5_img-size-64_epoch400_20241216.pth
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--cuda', '-c', type=int, default=0, help='Number of GPUs to use')
    parser.add_argument('--discrete', '-ds', type=int, default=0, help='whether the IRS is quantized')

    return parser.parse_args()


def initEnv(User_number=5,
            AP_antenna_number=1,
            IRS_scale=16,
            User_pos=None,):

    interval = 0.03
    IRS_pos = np.array([interval / 2, interval / 2, 0])
    AP_pos = np.array([IRS_scale * interval / 2, IRS_scale * interval / 2, 1.5])

    if User_pos is None:
        User_pos = []
        for i in range(User_number):
            rdn1 = (np.random.rand() * 2 - 1) * np.pi / 3
            rdn2 = (np.random.rand() * 2 - 1) * np.pi / 3
            height = np.random.randint(1, 2)
            User_pos.append([IRS_scale * interval / 2 + height * np.tan(rdn1),
                             IRS_scale * interval / 2 + height * np.tan(rdn2), height])

    chnl = Saleh_Valenzuela_Channel(IRS_scale, IRS_pos, AP_pos, User_number, AP_antenna_number)
    chnl.genRician(User_pos, K=10)
    User_pos = np.array(User_pos)
    U2R_channel = chnl.genU2R()
    U2R_channel = np.split(U2R_channel, user_number, axis=1)
    return U2R_channel, User_pos, chnl


if __name__ == '__main__':
    args = get_args()

    if args.cuda is not None:
        device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0')

    mode = args.mode
    if mode == 'SOMA':
        from codec.models.SOMANet_PE import DMANet
    elif mode == 'SOMA-DSCN':
        from codec.models.SOMANet_DSCN import DMANet

    interval = 0.03
    withIRS = not args.noIRS
    optimizeIRS = args.optIRS if withIRS else False
    IRS_scale = args.IRS_scale
    user_number = args.user_number
    IRS_pos = np.array([interval / 2, interval / 2, 0])
    AP_pos = np.array([IRS_scale * interval / 2, IRS_scale * interval / 2, 4.5])
    AP_antenna_number = args.AP_antenna
    SNR = list(np.random.randint(low=1, high=20, size=(user_number,)))  # initialize only (will be reset during training)

    envkwargs = {"userNum": user_number,
                 "antennaNum": AP_antenna_number,
                 "IRS_scale": IRS_scale,
                 "SNR": SNR,
                 "AP_pos": AP_pos,
                 "IRS_pos": IRS_pos,
                 "dynamic_userNum": False,
                 "dynamic_position": False,
                 }
    trainer = Trainer(**envkwargs)

    preset_pos = [[ 1.13003522, 0.50472761, 1.  ],
                  [-0.00699707, -0.21143006, 1. ],
                  [-1.09645732, -0.27572401, 1. ],
                  [ 0.19318778, 1.00622525, 1.  ],
                  [ 0.19803969, 0.0115971,  1.  ]]
    envs, usr_pos, chnl = initEnv(user_number, AP_antenna_number, IRS_scale, User_pos=preset_pos)
    trainer.Usr_pos = usr_pos
    if args.MinCSPP is None:
        compressed_channel = 128
    else:
        compressed_channel = round(args.MinCSPP * 128 * user_number)
    kwargs = {"envs": envs,
              "img_size": int(256*args.img_scale),
              "compressed_channel": compressed_channel,
              "P": 1,
              "withIRS": withIRS,
              "CSI_bound": 30,
              "optimizeIRS": optimizeIRS,
              "device": device}
    SemMA = DMANet(**kwargs)

    expname = (args.mode + '-exp-ver'
               + ("_withIRS" if withIRS else "_noIRS")
               + ("_optIRS" if args.optIRS else "_fixIRS")
               + ("_IRS-scale-" + str(args.IRS_scale) if withIRS else "")
               + ("_AP-" + str(args.AP_antenna) + "_Usr-" + str(args.user_number))
               + "_img-size-" + str(int(256 * args.img_scale))
               + ("fixLR" if args.fix_lr else "")
               + (f"_MinCSPP-{args.MinCSPP}" if args.MinCSPP is not None else "")
               )

    tmp_model = DMANet(**kwargs)
    print("Begin Training... " + expname)

    csi_list = ['3-4', '13-10']
    train(expname=expname,
          model=SemMA,
          temp_model=tmp_model,
          trainer=trainer,
          chnl=chnl,
          epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.lr,
          device=device,
          img_scale=args.img_scale,
          droprate=args.dr,
          amp=args.amp,
          pretrained=args.load,
          schedule_type=csi_list,
          q=args.discrete,
          )
