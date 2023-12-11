import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.1'

import random
import argparse
import yaml
import time
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
from utils import demix_track, demix_track_demucs, sdr, get_model_from_config, eval_sdr_score, compute_sdr, decompose_signal

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import warnings

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

def valid(model, opts, config, device, verbose=False):
    model.eval()
    all_mixtures_path = glob.glob(opts.valid_path + '/*/mixture.wav')
    print('Total mixtures: {}'.format(len(all_mixtures_path)))

    if verbose:
        print('Total mixtures: {}'.format(len(all_mixtures_path)))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    all_sdr = dict()
    for instr in config.training.instruments:
        all_sdr[instr] = []

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    for path in all_mixtures_path:
        
        mix, sr = sf.read(path)
        folder = os.path.dirname(path)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        if verbose:
            print('Song: {}'.format(os.path.basename(folder)))
            
        if opts.model_type == 'htdemucs' or opts.model_type == 'hdemucs' or opts.model_type == 'bs_hdemucs':
            # res = demix_track_demucs(config, model, mixture, device)
            res: dict[str, np.array] = decompose_signal(
                model=model,
                model_sample_rate=sr,
                signal=mixture,
                signal_sample_rate=sr,
                device=device,
                sources_list=instruments,
                listener=None,
                normalise=True,
            )
        elif opts.model_type == 'dtt_net':
            res = demix_track(config, model, mixture, device)
        else:
            res = demix_track(config, model, mixture, device)

        if 1:
            pbar_dict = {}
            references = []
            estimates = []

            for instr in instruments:

                if instr != 'other' or config.training.other_fix is False:
                    track, sr1 = sf.read(folder + '/{}.wav'.format(instr))
                else:
                    # other is actually instrumental
                    track, sr1 = sf.read(folder + '/{}.wav'.format('vocals'))
                    track = mix - track
                    
                references.append(track)
                estimates.append(res[instr])

                    # references = np.expand_dims(track, axis=0)
                    # estimates = np.expand_dims(res[instr].T, axis=0)
                    # sdr_val = sdr(references, estimates)[0]
                    # if verbose:
                    #     print(instr, res[instr].shape, sdr_val)
                    # all_sdr[instr].append(sdr_val)
                    # pbar_dict['sdr_{}'.format(instr)] = sdr_val
            
            references = np.array(references)
            estimates = np.array(estimates)
            sdr_scores = eval_sdr_score(references, estimates, sources_order=instruments, sample_rate=sr)

            for instr in instruments:
                sdr_val = sdr_scores[instr]
                if verbose:
                    print(instr, res[instr].shape, sdr_val)
                all_sdr[instr].append(sdr_val)
                pbar_dict['sdr_{}'.format(instr)] = sdr_val
            
            try:
                all_mixtures_path.set_postfix(pbar_dict)
            except Exception as e:
                pass

    sdr_avg = 0.0
    for instr in instruments:
        sdr_val = np.array(all_sdr[instr]).mean()
        print("Instr SDR {}: {:.4f}".format(instr, sdr_val))
        sdr_avg += sdr_val
    sdr_avg /= len(instruments)
    if len(instruments) > 1:
        print('SDR Avg: {:.4f}'.format(sdr_avg))
    return sdr_avg


def get_args_parser():
    # parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('--epoch', type=int, default=3)
    # parser.add_argument('--batch_size', type=int, default=256)
    # parser.add_argument('--root', type=str, default='./cifar')
    # parser.add_argument('--local_rank', type=int)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--port', type=int, default=2033)
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, demucs, segm_models, mel_band_roformer, bs_roformer, dtt_net")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--results_path", type=str, help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="dataset path. Can be several parameters.")
    parser.add_argument("--dataset_type", type=int, default=1, help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", type=str, help="validate path")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", type=bool, default=False, help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
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

    train_set = MSSDataset(
        config,
        opts.data_path,
        metadata_path=os.path.join(opts.results_path, 'metadata_{}.pkl'.format(opts.dataset_type)),
        dataset_type=opts.dataset_type,
    )

    valid_set = MSSDataset(
        config.valid,
        opts.valid_path,
        metadata_path=os.path.join(opts.results_path, 'metadata_{}.pkl'.format(config.valid.dataset_type)),
        dataset_type=config.valid.dataset_type,
    )

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    valid_sampler = DistributedSampler(dataset=valid_set, shuffle=False)
    
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, config.training.batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=opts.num_workers)

    batch_sampler_valid = torch.utils.data.BatchSampler(valid_sampler, config.valid.batch_size, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_sampler=batch_sampler_valid, num_workers=opts.num_workers)
    
    model = get_model_from_config(opts.model_type, config)

    model = model.cuda(local_gpu_id)
    model = DistributedDataParallel(module=model, device_ids=[local_gpu_id],find_unused_parameters=True)

    # if local_gpu_id == 0:
    #     wandb.watch(model)

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

    all_sdr = dict()
    for instr in instruments:
        all_sdr[instr] = []
    
    sr = 44100
    log = dict()

    log_header = ['epoch', 'train_loss', 'sdr_avg', 'sdr_scores']
    for header in log_header:
        log[header] = []

    for epoch in range(config.training.num_epochs):
        model.train()
        loss_avg = 0.
        total = 0
        train_sampler.set_epoch(epoch)

        print('Train epoch: {} Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))

        pbar = tqdm(train_loader)
        for i, (batch, mixes) in enumerate(pbar):
            y = batch.to(local_gpu_id)
            x = mixes.to(local_gpu_id)

            if config.training.source_weights is not None:
                    weight = torch.tensor(config.training.source_weights).to(y)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if opts.model_type == 'hdemucs' or opts.model_type == 'htdemucs' or opts.model_type == 'bs_hdemucs':
                    dims = tuple(range(2, y.dim()))
                    y_ = model.forward(x)
                    loss = criterion(y_, y)
                    loss = loss.mean(dims).mean(0)
                    loss = (loss * weight).sum() / weight.sum()
                
            loss /= gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            if config.training.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            li = loss.item() * gradient_accumulation_steps
            loss_avg += li
            total += 1
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_avg / (i + 1)})
            loss.detach()

        print('Training loss: {:.6f}'.format(loss_avg / total), "epoch: ", epoch)
        print("Valid Start")

        pbar = tqdm(valid_loader)
        sdr_avg = 0.0
        loss_val = 0.
        
        model.eval()
        with torch.no_grad():

            for i, (batch, mixes) in enumerate(pbar):
                y = batch.to(local_gpu_id)
                x = mixes.to(local_gpu_id)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    if opts.model_type == 'htdemucs' or opts.model_type == 'hdemucs' or opts.model_type == 'bs_hdemucs':

                        # res = demix_track_demucs(config, model, mixture, device)
                        res: dict[str, np.array] = decompose_signal(
                            model=model,
                            model_sample_rate=sr,
                            signal=x,
                            signal_sample_rate=sr,
                            device=local_gpu_id,
                            sources_list=instruments,
                            listener=None,
                            normalise=True,
                        )
                pbar_dict = {}

                est_stack = np.stack([res[inst] for inst in instruments])
                est_stack = torch.from_numpy(est_stack).unsqueeze(0).permute(0, 1, 3, 2).to(local_gpu_id)

                sdr_scores = compute_sdr(y, est_stack).squeeze(0)
                
                for i, instr in enumerate(instruments):
                    sdr_val = sdr_scores[i]
                    all_sdr[instr].append(sdr_val)
                    pbar_dict['sdr_{}'.format(instr)] = float(sdr_val)
                
                pbar_dict['sdr_avg'] = float(sdr_scores.mean())
                
                try:
                    pbar.set_postfix(pbar_dict)
                except Exception as e:
                    pass

        sdr_avg = 0.0
        for instr in instruments:
            sdr_val = torch.Tensor(all_sdr[instr]).cpu().numpy().mean()
            print("Instr SDR {}: {:.4f}".format(instr, sdr_val))
            sdr_avg += sdr_val

        sdr_avg /= len(instruments)
        print('SDR Avg: {:.4f} of epoch {}'.format(sdr_avg, epoch))

        log['epoch'].append(epoch)
        log['train_loss'].append(loss_avg / total)
        log['sdr_avg'].append(sdr_avg)
        log['sdr_scores'].append(all_sdr)

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

        with open(os.path.join(opts.results_path, 'logs','log.txt'), 'a') as f:
                f.write(str(log) + '\n')

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


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser('Distributed training test', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.ngpus_per_node = torch.cuda.device_count()
    opts.gpu_ids = list(range(opts.ngpus_per_node))
    opts.num_workers = opts.ngpus_per_node * 4

    torch.multiprocessing.spawn(main,
             args=(opts,),
             nprocs=opts.ngpus_per_node,
             join=True)

    # wandb.finish()