# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from torchinfo import summary
from torchviz import make_dot

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from collections import namedtuple

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset, build_dataset_test
from engine import train_one_epoch_st,test_st, test_st_val
from models_istt import build_model
import logging
import pprint
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm
import numpy as np
import random

import psutil
import os
import wandb
import pickle as pkl

from util.misc import nested_tensor_from_tensor_list

# def monitor_cpu():
#     pid = os.getpid()
#     process = psutil.Process(pid)
#     while True:
#         cpu_cores = process.cpu_num()  # Текущее ядро
#         cpu_percent = process.cpu_percent(interval=1) / psutil.cpu_count()
#         print(f"Используется ядро: {cpu_cores}, Нагрузка: {cpu_percent:.1f}%")

# # Запустить в отдельном потоке
# import threading

def load_state_dict_custom(model, state_dict):
    model_state = model.state_dict()
    filtered_state = {
        k: v 
        for k, v in state_dict.items() 
        if k in model_state and v.size() == model_state[k].size()
    }
    model.load_state_dict(filtered_state, strict=False)
    return model


class WarmupThenPlateau:
    def __init__(self, optimizer, warmup_steps, base_lr, plateau_config):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
        self.plateau_scheduler = ReduceLROnPlateau(optimizer, **plateau_config)
        
    def step(self, metrics=None):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr_scale = min(1., float(self.current_step) / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_scale * self.base_lr
        elif metrics is not None:
            self.plateau_scheduler.step(metrics)
    
    def state_dict(self):
        return {
            'warmup_steps': self.warmup_steps,
            'base_lr': self.base_lr,
            'current_step': self.current_step,
            'plateau_scheduler': self.plateau_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict['warmup_steps']
        self.base_lr = state_dict['base_lr']
        self.current_step = state_dict['current_step']
        self.plateau_scheduler.load_state_dict(state_dict['plateau_scheduler'])


def set_global_seed(seed: int) -> None:
    """
    Set global seed for reproducibility.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# print(torch.version.cuda)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--run_name', default='', type=str,
                        help="Name of run in wandb")
    parser.add_argument('--in_content_folder', default="", type=str)
    parser.add_argument('--style_folder',default="", type=str)
    parser.add_argument('--in_content_folder_val', default="", type=str)
    parser.add_argument('--style_folder_val',default="", type=str)
    parser.add_argument('--target_folder', default="", help='path to target images', type=str)
    
    
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--num_warmup', default=100, type=int)
    
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    
    parser.add_argument('--epochs', default=40, type=int)
    
    parser.add_argument('--accum_steps', default=1, type=int)
    parser.add_argument('--save_image_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=5, type=float, help='gradient clipping max norm') ###

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    
    #################### jianbo
    parser.add_argument('--model_type', default='nofold', type=str,
                        help="type of model")
    parser.add_argument('--fold_k', default=8, type=int,
                        help="Size of fold kernels")
    parser.add_argument('--fold_stride', default=6, type=int,
                        help="Size of fold kernels")
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--enorm', action='store_true')
    parser.add_argument('--dnorm', action='store_true')
    parser.add_argument('--tnorm', action='store_true')
    parser.add_argument('--model_pre', action='store_true')
    parser.add_argument('--cbackbone_layer', type=int, default=4,
                        help="")
    parser.add_argument('--sbackbone_layer', type=int, default=4,
                        help="")

    # No weights
    parser.add_argument('--no_weights', action='store_true',
                        help='if we need to init from random weights')

    # # Init start weights
    # parser.add_argument('--start_weights', action='store_true',
    #                     help='if we need to init from start weights')
    
    # Testing / Training
    parser.add_argument('--testing', action='store_true',
                        help='test')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--distill_loss_type', default='MSE', type=str,
                        help="loss type for distillation")
    
    # parser.add_argument('--dice_loss_coef', default=1, type=float)
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    
    parser.add_argument('--content_loss_coef', default=1.0, type=float)
    parser.add_argument('--style_loss_coef', default=1e4, type=float)
    parser.add_argument('--tv_loss_coef', default=0, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--train_size', default=0.8, type=float)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--wikiart_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default="output_nofold_fix512",
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args, contents_path, styles_path):
    # thread = threading.Thread(target=monitor_cpu)
    # thread.start()
    
    # # Ограничение числа потоков для операций внутри PyTorch
    # os.environ["OMP_NUM_THREADS"] = "1"         # Для операций OpenMP
    # os.environ["MKL_NUM_THREADS"] = "1"        # Для Intel MKL
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"   # Для OpenBLAS (если используется)
    # torch.set_num_threads(1)          # Основные операции
    # torch.set_num_interop_threads(1)  # Межпоточные операции (редко нужно)
    
    utils.init_distributed_mode(args)
    
    output_dir = Path(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    logging.basicConfig(filename=os.path.join(output_dir, "log_eval.txt"),
                    format='%(asctime)-15s %(message)s')
        
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    logger.info(pprint.pformat(args))
    
    
    device = torch.device(args.device)
    # device = torch.device('cpu')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    set_global_seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # logger.info(pprint.pformat(model))
    
    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('number of params:', n_parameters)
    logger.info('number of params: {}'.format(n_parameters))
    
    testing = args.testing
    no_weigths = args.no_weights
    # start_weights = args.start_weights

    cnt_inits = 0

    if no_weigths:
        with torch.no_grad():
            for name, params in model.named_parameters():
                if ('weight' in name):
                    torch.nn.init.normal_(params, mean=0.0, std=0.02)
                    cnt_inits += 1
                elif ('bias' in name):
                    torch.nn.init.constant_(params, 0.0)
                    cnt_inits +=1

        print('Количество инициализаций:', cnt_inits)
                
    # param_dicts = [
    #     {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    # ]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.4, patience=2, )
    lr_scheduler = WarmupThenPlateau(
        optimizer,
        warmup_steps=args.num_warmup,
        base_lr=args.lr,
        plateau_config={
            'mode': 'min',
            'factor': 0.4,
            'patience': 2,
            'verbose': False
        }
    )
    
    # if start_weights:
    #     checkpoint = {}
    #     checkpoint['epoch'] = 1
        
    #     with open('./weights_dict.pkl', 'rb') as file:
    #         d_w = pkl.load(file)
            
    #     for name, param in model.named_parameters():
    #         if (name in d_w) and (param.data.shape == d_w[name].data.shape):
    #             with torch.no_grad():
    #                 param.data = d_w[name].data.to(param.device)
    
    # # Параметры warmup
    # warmup_steps = 20
    # initial_lr = 1e-6  # Начальный lr для warmup
    # base_lr = 1.0    # Основной lr после warmup
    
    # # 1. Создаем лямбда-функцию для линейного warmup
    # def warmup_lambda(step):
    #     if step < warmup_steps:
    #         return initial_lr + (base_lr - initial_lr) * (step / warmup_steps)
    #     return 1.0  # После warmup возвращаем множитель 1.0
    
    # # 2. Warmup-шедулер
    # warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    if testing:
        # Testing
        dataset_val = build_dataset_test('val', args) 
    
        print('Путь к контентам: ', contents_path, 'Путь к стилям: ', styles_path)
        print('Длина датасета: ', len(dataset_val))
        # Для модели с input_shape=(batch, channels, height, width)
        # Если модель принимает два тензора:
        # input_size = [(1, 3, 512, 512), (1, 3, 512, 512)]  # shapes для content и style изображений
        # summary(model, input_size=input_size)
        # Переводим модель в режим оценки (отключает Dropout, BatchNorm и т.д.)
        
        # for name, params in model.named_parameters():
        #     if not params.requires_grad:
        #         params.requires_grad = True
        # for name, params in model.named_parameters():
        #     print(name)
        # exit()
        # if args.distributed:
        #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
        # else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn_st_test, num_workers=args.num_workers)

        if (not no_weigths):
            checkpoint = torch.load(args.resume, map_location='cuda')
            model = load_state_dict_custom(model, checkpoint['model'])
    
        test_stats = test_st_val(model, criterion, postprocessors, data_loader_val, device, logger, checkpoint['epoch'], str(output_dir))
    else:
        # Training
        run = wandb.init(project="science", name=f'exp_time_{str(time.time())}' + args.run_name, config=args)
        
        dataset_train = build_dataset('train', args)
        
        set_global_seed(42)
        generator1 = torch.Generator().manual_seed(42)
        dataset_train, dataset_eval = random_split(dataset_train, [int(len(dataset_train) * args.train_size), len(dataset_train) - int(len(dataset_train) * args.train_size)], generator=generator1)
        # set_global_seed(42)
        # generator2 = torch.Generator().manual_seed(42)
        # dataset_eval, dataset_test = random_split(dataset_eval, [int(len(dataset_eval) * 0.5), len(dataset_eval) - int(len(dataset_eval) * 0.5)], generator=generator2)
        print('Длина train сета:', len(dataset_train))
        print('Длина eval сета:', len(dataset_eval))
        # print('train first element:', dataset_train[0][2])
        # print('eval first element:', dataset_eval[0][2])
        
        
        # if args.distributed:
        #     sampler_eval = DistributedSampler(dataset_val, shuffle=False)
        #     sampler_train = DistributedSampler(dataset_train, shuffle=False)
        # else:
        sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    

        data_loader_train = DataLoader(dataset_train, args.batch_size, sampler=sampler_train,
                                     drop_last=False, collate_fn=utils.collate_fn_st, num_workers=args.num_workers)
        data_loader_eval = DataLoader(dataset_eval, args.batch_size, sampler=sampler_eval,
                                     drop_last=False, collate_fn=utils.collate_fn_st, num_workers=args.num_workers)

        if (not no_weigths):
            checkpoint = torch.load(args.resume, map_location='cuda')
            model = load_state_dict_custom(model, checkpoint['model'])

        # d_w = {}
        # for name, param in model.named_parameters():
        #     d_w[name] = param
        # with open('./weights_dict.pkl', 'wb') as file:
        #     pkl.dump(d_w, file)
        # exit()
        
        for name, params in model.named_parameters():
            if not params.requires_grad and (args.lr_backbone > 0 or "backbone" not in name):
                params.requires_grad = True
                
        for name, params in model.named_parameters():
            if not params.requires_grad:
                print("\033[92m Слой "+name+' не учится!\033[0m')
        
        # Train process
        it = 1
        global_loss=[-1]
        for epoch in range(1, args.epochs + 1):
            train_results, it = train_one_epoch_st(model, criterion, data_loader_train, data_loader_eval, optimizer, lr_scheduler, 'cuda', logger, postprocessors, output_dir, epoch, './checkpoint_model_1', args, 0, it, global_loss, run)
            print('Epoch results:', train_results)
            # print('\033[95mEVAL Epoch results:', test_st(model, criterion, postprocessors, data_loader_eval, 'cuda', logger, epoch, str(output_dir)),'\033[0m')

        
        
if __name__ == '__main__':
    
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    
    args = parser.parse_args([
          "--batch_size", "4",
          '--content_loss_coef', "1",
          "--style_loss_coef", "10",
          "--fold_k", "5",
          "--lr_backbone", "1e-5",
          "--lr", "1e-5",
          "--num_warmup", "150",
          "--enc_layers", "10",
          "--dec_layers", "10",
          "--model_type", "nofold",
          "--enorm",
          "--dnorm", 
          "--tnorm",
          "--cbackbone_layer", "2",
          "--sbackbone_layer", "4",
          "--dataset_file", "demo",
          "--resume", "checkpoint_model/checkpoint0005.pth",
          "--img_size", "512",
          "--in_content_folder", "inputs/content",
          "--style_folder", "inputs/style",
          "--in_content_folder_val", "inputs/content_test",
          "--style_folder_val", "inputs/style_test",
          # "--in_content_folder","inputs/content_test",
          # "--style_folder","inputs/style_test",
          "--output_dir", "outputs",
          "--target_folder", "inputs/target",
          "--train_size", "0.95",
          "--epochs", "8",
        
          "--accum_steps", "20",
          "--save_image_freq", "160",
          "--save_freq", "60",
          "--eval_freq", "30",

          # "--run_name", "author_sw_lback_c2_s4_tr_3_3_lambda_10",
          "--run_name", "gatys_sw_lback_c2_s4_tr_6_6",
          "--distill_loss_type", "Gatys", # distill loss type
          # "--no_weights", # Normal initx
     
          "--testing", # train or test
    ])
    
    main(args, args.in_content_folder, args.style_folder)

    # # Выводим максимальное число занятых байтов
    # max_memory=torch.cuda.max_memory_allocated()
    # msg='max_mem: {})'.format(max_memory)
    # print(msg)
    # max_memory = torch.cuda.max_memory_allocated()  # Макс. выделено (в байтах)

# # Выводим затраты по GPU
# print(f"\033[93mMax GPU memory used: {max_memory / 1024**2:.2f} MB\033[0m")
