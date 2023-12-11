import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.1'

import warnings
warnings.filterwarnings("ignore")

import json
import random
import argparse
import yaml
import time
from pathlib import Path
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from dataset import MSSDataset
from utils import demix_track, demix_track_demucs, sdr, get_model_from_config, eval_sdr_score, compute_sdr
from evaluate import (
    apply_gains,
    apply_ha,
    make_scene_listener_list,
    remix_stems,
    load_reference_stems
)
from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal
from clarity.utils.source_separation_support import get_device, separate_sources
from parallel import DataParallelModel, DataParallelCriterion
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import warnings

from clarity.utils.signal_processing import (
    clip_signal,
    denormalize_signals,
    normalize_signal,
    resample,
    to_16bit,
)

from valid import valid

def decompose_signal(
    model: torch.nn.Module,
    model_sample_rate: int,
    signal: np.array,
    signal_sample_rate: int,
    device: torch.device,
    sources_list,
    listener: Listener,
    normalise: bool = True,
):
    """
    Decompose signal into 8 stems.

    The listener is ignored by the baseline system as it
     is not performing personalised decomposition.
    Instead, it performs a standard music decomposition using a pre-trained
     model trained on the MUSDB18 dataset.

    Args:
        model (torch.nn.Module): Torch model.
        model_sample_rate (int): Sample rate of the model.
        signal (ndarray): Signal to be decomposed.
        signal_sample_rate (int): Sample frequency.
        device (torch.device): Torch device to use for processing.
        sources_list (list): List of strings used to index dictionary.
        listener (Listener).
        normalise (bool): Whether to normalise the signal.

     Returns:
         Dictionary: Indexed by sources with the associated model as values.
    """
    if signal.shape[0] > signal.shape[1]:
        signal = signal.T

    if signal_sample_rate != model_sample_rate:
        signal = resample(signal, signal_sample_rate, model_sample_rate)

    if normalise:
        signal, ref = normalize_signal(signal)

    sources = separate_sources(
        model,
        torch.from_numpy(signal.astype(np.float32)),
        model_sample_rate,
        device=device,
    )
    # # only one element in the batch
    sources = sources[0]

    if normalise:
        sources = denormalize_signals(sources, ref)

    sources = np.transpose(sources, (0, 2, 1))
    return dict(zip(sources_list, sources))


def masked_loss(y_, y, q, coarse=True):
    # shape = [num_sources, batch_size, num_channels, chunk_size]
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()


def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--port', type=int, default=2033)
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, demucs, segm_models, mel_band_roformer, bs_roformer, dtt_net")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--results_path", type=str, help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="dataset path. Can be several parameters.")
    parser.add_argument("--dataset_type", type=int, default=1, help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", type=str, help="validate path")
    parser.add_argument("--pin_memory", type=bool, default=False, help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--use_mse_loss", action='store_true', help="Use default MSE loss")
    parser.add_argument("--use_l1_loss", action='store_true', help="Use L1 loss")
    parser.add_argument("--our_model", type=str, default='', help="use our model", required=False)
    return parser

def init_distributed_training(rank, opts):
    # 1. setting for distributed training
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    torch.distributed.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:' + str(opts.port),
                            world_size=opts.ngpus_per_node,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()

    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print('opts :',opts)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main(rank, opts):
    init_distributed_training(rank, opts)
    local_gpu_id = opts.gpu

    manual_seed(opts.seed + int(time.time()))
    torch.backends.cudnn.benchmark = True

    with open(opts.config_path) as f:
        if opts.model_type == 'htdemucs' or opts.model_type == 'hdemucs' or opts.model_type == 'bs_hdemucs':
            config = OmegaConf.load(opts.config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    print("Instruments: {}".format(config.training.instruments))
    
    if not os.path.isdir(opts.results_path):
        os.mkdir(opts.results_path)

    use_amp = True
    try:
        use_amp = config.training.use_amp
    except:
        pass

    word_size = 2
    node = 1
    rank = 0

    t_easy = config.dataset.train_easy
    t_all = config.dataset.train_all

    easy_trainset = MSSDataset(
        config,
        opts.data_path,
        metadata_path=os.path.join(opts.results_path, 'metadata_trainEASY.pkl'),
        dataset_type=opts.dataset_type,
        cl_path=t_easy
        )
    
    all_trainset = MSSDataset(
        config,
        opts.data_path,
        metadata_path=os.path.join(opts.results_path, 'metadata_trainALL.pkl'),
        dataset_type=opts.dataset_type,
        cl_path=t_all
        )
    
    easy_train_sampler = DistributedSampler(dataset=easy_trainset, shuffle=True)
    all_train_sampler = DistributedSampler(dataset=all_trainset, shuffle=True)

    easy_batch_sampler_train = torch.utils.data.BatchSampler(easy_train_sampler, config.training.batch_size, drop_last=True)
    all_batch_sampler_train = torch.utils.data.BatchSampler(all_train_sampler, config.training.batch_size, drop_last=True)

    easy_trainloader = DataLoader(
        easy_trainset,
        batch_sampler=easy_batch_sampler_train,
        #batch_size=config.training.batch_size,
        #shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=opts.pin_memory)
    
    all_trainloader = DataLoader(
        all_trainset,
        batch_sampler=all_batch_sampler_train,
        #batch_size=config.training.batch_size,
        #shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=opts.pin_memory)

    print('Train samples: easy {}, all {}'.format(len(easy_trainset), len(all_trainset)))
    
    model = get_model_from_config(opts.model_type, config)
    model = model.cuda(local_gpu_id)
    model = DistributedDataParallel(module=model, device_ids=[local_gpu_id],find_unused_parameters=True)

    if opts.start_check_point != '':
        load_not_compatible_weights(model, opts.start_check_point, verbose=True)
        print('Load weights: {}'.format(opts.start_check_point))

    if opts.model_type == 'hdemucs' or opts.model_type == 'htdemucs' or opts.model_type == 'bs_hdemucs':
        criterion = nn.L1Loss(reduction='none').to(local_gpu_id)

    if config.training.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr)
    elif config.training.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr)
    elif config.training.optimizer == 'sgd':
        print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, momentum=0.999)
    else:
        print('Unknown optimizer: {}'.format(config.training.optimizer))
        exit()

    gradient_accumulation_steps = int(config.training.gradient_accumulation_steps)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience, factor=config.training.reduce_factor)

    scaler = GradScaler()
    print('Train for: {}'.format(config.training.num_epochs))
    best_sdr = -100

    instruments = config.training.instruments
    sr = 44100
    alpha_t = 0.8
    y_1 = None

    # sdr_avg = valid(model, opts, config, device=local_gpu_id, verbose=False)
    # print("SDR FIRST:", sdr_avg)

    train_loader = easy_trainloader

    for epoch in range(config.training.num_epochs):
        model.train()
        print('Train epoch: {} Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        loss_val = 0.
        total = 0

        alpha_t = alpha_t * ((epoch + 1) / config.training.num_epochs)
        alpha_t = max(0, alpha_t)
        
        if epoch < 30:
            easy_train_sampler.set_epoch(epoch)
        else:
            all_train_sampler.set_epoch(epoch)
        
        if epoch == 30:
            train_loader = all_trainloader
            best_sdr = 0
            print("Train on all")
            
        pbar = tqdm(train_loader)
        for i, (batch, mixes) in enumerate(pbar):
            y = batch.to(local_gpu_id)
            x = mixes.to(local_gpu_id)  # mixture

            if config.training.source_weights is not None:
                weight = torch.tensor(config.training.source_weights).to(y)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if opts.model_type == 'hdemucs' or opts.model_type == 'htdemucs' or opts.model_type == 'bs_hdemucs':
                    dims = tuple(range(2, y.dim()))
                    y_ = model.forward(x)

                    if epoch == 0:
                        y_1 = y_

                    soft_targets = ((1 - alpha_t) * y) + (alpha_t * y_1)
                    soft_targets = soft_targets.cuda()

                    loss = criterion(y_, soft_targets)
                    loss = loss.mean(dims).mean(0)
                    loss = (loss * weight).sum() / weight.sum()

                    y_1 = y_
                
            loss /= gradient_accumulation_steps

            scaler.scale(loss).backward(retain_graph=True)
            if config.training.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            li = loss.item() * gradient_accumulation_steps
            loss_val += li
            total += 1
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
            loss.detach()

        print('Training loss: {:.6f}'.format(loss_val / total),"epoch: ", epoch)

        sdr_avg = valid(model, opts, config, device=local_gpu_id, verbose=False, epoch=epoch)

        if sdr_avg > best_sdr and sdr_avg > 0:
            store_path = opts.results_path + '/ckpt/' + '/model_{}_ep_{}_sdr_{:.4f}.ckpt'.format(opts.model_type, epoch, sdr_avg)
            print('Store weights: {}'.format(store_path))
            state_dict = model.state_dict() if type(local_gpu_id) == int else model.module.state_dict()
            torch.save(
                state_dict,
                store_path
            )
            best_sdr = sdr_avg
        scheduler.step(sdr_avg)

        # Save last
        store_path = opts.results_path + '/ckpt/' + '/last_{}.ckpt'.format(opts.model_type)
        state_dict = model.state_dict() if type(local_gpu_id) == int else model.module.state_dict()
        torch.save(
            state_dict,
            store_path
        )

def load_not_compatible_weights(model, weights, verbose=False):
    new_model = model.state_dict()
    old_model = torch.load(weights)

    for el in new_model:
        if el in old_model:
            if verbose:
                print('Match found for {}!'.format(el))
            if new_model[el].shape == old_model[el].shape:
                if verbose:
                    print('Action: Just copy weights!')
                new_model[el] = old_model[el]
            else:
                if len(new_model[el].shape) != len(old_model[el].shape):
                    if verbose:
                        print('Action: Different dimension! Too lazy to write the code... Skip it')
                else:
                    if verbose:
                        print('Shape is different: {} != {}'.format(tuple(new_model[el].shape), tuple(old_model[el].shape)))
                    ln = len(new_model[el].shape)
                    max_shape = []
                    slices_old = []
                    slices_new = []
                    for i in range(ln):
                        max_shape.append(max(new_model[el].shape[i], old_model[el].shape[i]))
                        slices_old.append(slice(0, old_model[el].shape[i]))
                        slices_new.append(slice(0, new_model[el].shape[i]))
                    # print(max_shape)
                    # print(slices_old, slices_new)
                    slices_old = tuple(slices_old)
                    slices_new = tuple(slices_new)
                    max_matrix = np.zeros(max_shape, dtype=np.float32)
                    for i in range(ln):
                        max_matrix[slices_old] = old_model[el].cpu().numpy()
                    max_matrix = torch.from_numpy(max_matrix)
                    new_model[el] = max_matrix[slices_new]
        else:
            if verbose:
                print('Match not found for {}!'.format(el))
    model.load_state_dict(
        new_model
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser('Distributed training curri', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.ngpus_per_node = torch.cuda.device_count()
    opts.gpu_ids = list(range(opts.ngpus_per_node))
    opts.num_workers = opts.ngpus_per_node * 4

    torch.multiprocessing.spawn(main,
             args=(opts,),
             nprocs=opts.ngpus_per_node,
             join=True)