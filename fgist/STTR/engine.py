# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import time
import math
import os
import sys
from typing import Iterable

import torch
import torch.nn as nn
import util.misc as utils

from torchvision.utils import save_image

from datasets.demo import denorm
import shutil
import pandas as pd
import tqdm
# from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import kornia.metrics as metrics

# ----------------------------------------------------------------------------------
def train_one_epoch_st(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, data_loader_eval: Iterable, optimizer: torch.optim.Optimizer,lr_scheduler,
                    device: torch.device,logger, postprocessors, output_dir, epoch: int, save_path:str, args, max_norm: float = 0, new_it = 0, global_loss=None, run=None):
    model.train()
    criterion.train()
    optimizer.zero_grad()
    if args.distill_loss_type == 'author':
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    gradient_accumulation_steps = args.accum_steps
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    run.log({"Epoch num": epoch})
    
    save_image_freq = args.save_image_freq # 80
    save_freq = args.save_freq # 10
    eval_freq = args.eval_freq # 30
    
    sum_loss = 0
    
    for it, (samples, style_images, targets, target_images) in tqdm.tqdm(enumerate(data_loader, 1), total=len(data_loader)):
        
        samples = samples.to(device)
        style_images = style_images.to(device)
        target_images = target_images.to(device)
        # target_images.tensors = target_images.tensors * 0
        
        # outputs = model(samples, style_images)    
        # print(outputs.shape, target_images.tensors.shape, style_images.tensors.shape)
        if args.distill_loss_type == 'author':
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                outputs = model(samples, style_images) 
                loss_dict = criterion(outputs, target_images, samples, style_images)
        else:
            outputs = model(samples, style_images) 
            loss_dict = criterion(outputs, target_images, samples, style_images)
            
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            logger.info(loss_dict_reduced)
            sys.exit(1)
        if it % save_image_freq == 0:
            # print('СОХРАНЯЕТ')
            if not os.path.exists(os.path.join(save_path,"train_outputs")):
                os.makedirs(os.path.join(save_path,"train_outputs"))
            if not os.path.exists(os.path.join(save_path,"train_content_images")):
                os.makedirs(os.path.join(save_path,"train_content_images"))
            if not os.path.exists(os.path.join(save_path,"train_target_images")):
                os.makedirs(os.path.join(save_path,"train_target_images"))
            if not os.path.exists(os.path.join(save_path,"train_style_images")):
                os.makedirs(os.path.join(save_path,"train_style_images"))
    
            if isinstance(outputs, tuple):
                outputs,_=outputs
            outputs=denorm(outputs, device)
            samples.tensors=denorm(samples.tensors, device)
            style_images.tensors=denorm(style_images.tensors, device)
            target_images.tensors=denorm(target_images.tensors, device) 
            
            save_image(outputs,os.path.join(save_path,"train_outputs",f'{epoch:04}_{it:08}.png')  )   
            save_image(samples.tensors,os.path.join(save_path,"train_content_images",f'{epoch:04}_{it:08}.png' )  )  
            save_image(style_images.tensors,os.path.join(save_path,"train_style_images",f'{epoch:04}_{it:08}.png' )  )     # .clamp(0,1)
            save_image(target_images.tensors,os.path.join(save_path,"train_target_images",f'{epoch:04}_{it:08}.png' )  )
        # optimizer.zero_grad()
        if args.distill_loss_type == 'author':
            scaler.scale(losses).backward()
        else:
            losses.backward()
        # optimizer.step()
        
        if max_norm > 0:
            if args.distill_loss_type == 'author':
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        sum_loss += loss_value
        
        if  it % gradient_accumulation_steps == 0:
            if args.distill_loss_type == 'author':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            run.log({"Train_loss": sum_loss / gradient_accumulation_steps})
            print('Train_loss:', sum_loss / gradient_accumulation_steps)
            sum_loss = 0
            lr_scheduler.step()
            print('lr:', optimizer.param_groups[0]["lr"])

        if it % save_freq == 0:
            result = test_st(model, criterion, postprocessors, data_loader_eval, 'cuda', logger, epoch, str(output_dir), args)
            lr_scheduler.step(result['loss'])
            
            print('\033[95mEVAL Epoch results:', result,'\033[0m')
            print('Last lr:', optimizer.param_groups[0]["lr"])
            
            run.log({"Last lr": optimizer.param_groups[0]["lr"]})
            run.log({"Loss val": result['loss']})
            run.log({"MSE val": result['MSE'].item()})
            run.log({"MAE val": result['MAE'].item()})
            run.log({"Mean error per pixel in one channel": result['MAE'].item() / 512 / 512 / 3})
            run.log({"SSIM val": result['SSIM'].item()})
            
            model.train()
            criterion.train()
            
            if  global_loss[0] == -1 or global_loss[0] > result['loss']:
                global_loss[0] = result['loss']
                
                if not os.path.exists(os.path.join(save_path,"checkpoint")):
                    os.makedirs(os.path.join(save_path,"checkpoint"))
                checkpoint_path = os.path.join(save_path,"checkpoint",f'checkpoint{epoch:04}.pth')
        
                utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                }, checkpoint_path)

        torch.cuda.empty_cache()
        new_it += 1

    optimizer.step()
    optimizer.zero_grad()
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:{}".format(metric_logger))
    return ({k: meter.global_avg for k, meter in metric_logger.meters.items()}, new_it)

def save_batch_images(save_path,outputs,samples,style_images,epoch,it,device):
    
    outputs=denorm(outputs, device)
    samples.tensors=denorm(samples.tensors, device)
    style_images.tensors=denorm(style_images.tensors, device)
#             print("outputs.shape:",outputs.shape)
    save_image(outputs,os.path.join(save_path,"test_outputs",f'{epoch:04}',f'{epoch:04}_{it:08}.png' )  )  
    save_image(samples.tensors,os.path.join(save_path,"test_content_images",f'{epoch:04}',f'{epoch:04}_{it:08}.png' )  )  
    save_image(style_images.tensors,os.path.join(save_path,"test_style_images",f'{epoch:04}',f'{epoch:04}_{it:08}.png' )  ) 

@torch.no_grad()
def test_st(model, criterion, postprocessors, data_loader,  device, logger, epoch, save_path, args):
    model.eval()
    criterion.eval()

    save_image_freq = args.save_image_freq

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'
    new_it = 0
    
    if os.path.exists(os.path.join(save_path,"test_outputs",f'{epoch:04}')):
        shutil.rmtree(os.path.join(save_path,"test_outputs",f'{epoch:04}'))
    os.makedirs(os.path.join(save_path,"test_outputs",f'{epoch:04}'))

    metrics_result = {'MSE':0, 'MAE':0, 'SSIM':0}
    MSE_obj = nn.MSELoss(reduction='sum')
    MAE_obj = nn.L1Loss(reduction='sum')
    
    batch_size = args.batch_size
    cnt_all_objects = 0
    
    tmp_out=[]
    for it, (samples, style_images, targets, target_images) in tqdm.tqdm(metric_logger.log_every(data_loader, 300, logger, header), total=len(data_loader)):

        cnt_all_objects += samples.tensors.shape[0]
        
        samples = samples.to(device)
        style_images = style_images.to(device)
        target_images = target_images.to(device) 

        if args.distill_loss_type == 'author':
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                with torch.inference_mode():
                    outputs = model(samples, style_images) 
                    loss_dict = criterion(outputs, target_images, samples, style_images)
        else:
            outputs = model(samples, style_images) 
            loss_dict = criterion(outputs, target_images, samples, style_images)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        outputs=denorm(outputs, device)
        samples.tensors=denorm(samples.tensors, device)
        style_images.tensors=denorm(style_images.tensors, device)
        target_images.tensors=denorm(target_images.tensors, device)

        
        metrics_result['MSE'] += MSE_obj(outputs, target_images.tensors)
        metrics_result['MAE'] += MAE_obj(outputs, target_images.tensors)
        # metrics_result['SSIM'] += SSIM(data_range=1.0 if target_images.tensors.max() <= 1 else 255.0)
        metrics_result['SSIM'] += metrics.ssim(outputs, target_images.tensors, window_size=11).mean()

        
        if new_it % save_image_freq == 0:
            if not os.path.exists(os.path.join(save_path,"test_content_images",f'{epoch:04}')):
                os.makedirs(os.path.join(save_path,"test_content_images",f'{epoch:04}'))
            if not os.path.exists(os.path.join(save_path,"test_style_images",f'{epoch:04}')):
                os.makedirs(os.path.join(save_path,"test_style_images",f'{epoch:04}'))  
            if not os.path.exists(os.path.join(save_path,"test_target_images",f'{epoch:04}')):
                os.makedirs(os.path.join(save_path,"test_target_images",f'{epoch:04}'))   
          
            if isinstance(outputs, tuple):
                outputs, _ = outputs

            if "content_image_name" in targets[0]:
                for i in range(len(outputs)):
                    c_name=targets[i]["content_image_name"]
                    s_name=targets[i]["style_image_name"]
                    save_name="{}_{}".format(c_name,s_name)
                    
                    # output_i=denorm(outputs[i], device)
                    save_image(outputs[i], os.path.join(save_path,"test_outputs",f'{epoch:04}',f'{epoch:04}_{save_name}.png' )  )  
                    
                    # sample_i=denorm(samples.tensors[i], device)
                    save_image(samples.tensors[i], os.path.join(save_path,"test_content_images",f'{epoch:04}',f'{epoch:04}_{save_name}.png' )  )  
                    
                    # target_i=denorm(target_images.tensors[i], device)
                    save_image(target_images.tensors[i], os.path.join(save_path,"test_target_images",f'{epoch:04}',f'{epoch:04}_{save_name}.png' )  ) 

                    # style_i=denorm(style_images.tensors[i], device)
                    save_image(style_images.tensors[i], os.path.join(save_path,"test_style_images",f'{epoch:04}',f'{epoch:04}_{save_name}.png' )  ) 

        new_it += 1
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    metrics_result['MSE'] = metrics_result['MSE'] / cnt_all_objects * 255 * 255
    metrics_result['MAE'] = metrics_result['MAE'] / cnt_all_objects * 255
    metrics_result['SSIM'] /= len(data_loader)
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()} 
    stats.update(metrics_result)
    
    return stats


@torch.no_grad()
def test_st_val(model, criterion, postprocessors, data_loader,  device, logger, epoch, save_path):
    model.eval()
    criterion.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if os.path.exists(os.path.join(save_path,"test_outputs",f'{epoch:04}')):
        shutil.rmtree(os.path.join(save_path,"test_outputs",f'{epoch:04}'))
    os.makedirs(os.path.join(save_path,"test_outputs",f'{epoch:04}'))
    
    for it, (samples, style_images, targets) in metric_logger.log_every(data_loader, 100, logger, header):
        
        samples = samples.to(device)
        style_images = style_images.to(device)

        with torch.no_grad():
            outputs = model(samples, style_images)
            
        outputs = outputs.to(device)

        # reduce losses over all GPUs for logging purposes
        
        if it % 1 == 0:
            if not os.path.exists(os.path.join(save_path,"test_content_images",f'{epoch + 12000:04}')):
                os.makedirs(os.path.join(save_path,"test_content_images",f'{epoch + 12000:04}'))
            if not os.path.exists(os.path.join(save_path,"test_style_images",f'{epoch + 12000:04}')):
                os.makedirs(os.path.join(save_path,"test_style_images",f'{epoch + 12000:04}'))
            if not os.path.exists(os.path.join(save_path,"test_outputs",f'{epoch + 12000:04}')):
                os.makedirs(os.path.join(save_path,"test_outputs",f'{epoch + 12000:04}'))
            if isinstance(outputs, tuple):
                outputs,_=outputs
                
            if "content_image_name" in targets[0]:
                for i in range(len(outputs)):
                    c_name=targets[i]["content_image_name"]
                    s_name=targets[i]["style_image_name"]
                    save_name="{}_{}".format(c_name,s_name)
                    
                    output_i=denorm(outputs[i], device)
                    save_image(output_i,os.path.join(save_path,"test_outputs", f'{epoch + 12000:04}',f'{epoch:04}_{save_name}.png'))  
                    
                    sample_i=denorm(samples.tensors[i], device)
                    save_image(sample_i,os.path.join(save_path,"test_content_images", f'{epoch + 12000:04}',f'{epoch:04}_{save_name}.png'))       
            else:
                save_batch_images(save_path,outputs,samples,style_images,epoch,it,device)
    
    return {}
