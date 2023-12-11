# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

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

import warnings
warnings.filterwarnings("ignore")

from clarity.utils.results_support import ResultsFile

from utils import demix_track, demix_track_demucs, sdr, get_model_from_config, eval_sdr_score, decompose_signal


def valid(model, args, config, device, verbose=False, epoch=None):

    start_time = time.time()
    model.eval()
    all_mixtures_path = glob.glob(args.valid_path + '/*/mixture.wav')

    if epoch is not None:
        if epoch < 40:
            with open(config.dataset.valid_easy, 'r') as f:
                    easy_list = yaml.load(f, Loader=yaml.FullLoader)

            all_mixtures_path = [path for path in all_mixtures_path if path.split('/')[-2] in easy_list]
            print("Valid on Easy Samples:", len(all_mixtures_path))

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
        folder_name = os.path.abspath(folder)
        if verbose:
            print('Song: {}'.format(folder_name))
        # mixture = torch.tensor(mix.T, dtype=torch.float32)
        mixture = torch.tensor(mix.T, dtype=torch.float32)
        if args.model_type == 'htdemucs' or args.model_type == 'hdemucs':
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
        elif args.model_type == 'dtt_net':
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

                # if args.write_wav:
                #     sf.write("{}/{}_{}.wav".format(args.store_dir, os.path.basename(folder), instr), res[instr], sr, subtype='FLOAT')
                
                references.append(track)
                estimates.append(res[instr])

                # references = np.expand_dims(track, axis=0)
                # estimates = np.expand_dims(res[instr].T, axis=0)
                # sdr_val = sdr(references, estimates)[0]
                # if verbose:
                #     print(instr, res[instr].shape, sdr_val)
                # all_sdr[instr].append(sdr_val)
                # pbar_dict['sdr_{}'.format(instr)] = sdr_val
            try:
                references = np.array(references)
            except:
                # Make all arrays have the same shape as the first array
                first_shape = mix.shape

                for i in range(len(references)):
                    current_shape = references[i].shape
                    if current_shape != first_shape:
                        if current_shape[0] < first_shape[0]:
                            # If rows are fewer, extend the rows to match the first array
                            diff_rows = first_shape[0] - current_shape[0]
                            extension = np.random.rand(diff_rows, current_shape[1])
                            references[i] = np.vstack((references[i], extension))
                        elif current_shape[0] > first_shape[0]:
                            # If rows are more, trim the rows to match the first array
                            references[i] = references[i][:first_shape[0], :]

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


def check_validation(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, demucs, segm_models, mel_band_roformer, bs_roformer")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--valid_path", type=str, help="validate path")
    parser.add_argument("--store_dir", default="", type=str, help="path to store results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", type=bool, default=False, help="dataloader pin_memory")
    parser.add_argument("--our_model", type=bool, default=False, help="use our model", required=False)
    parser.add_argument("--write_wav", type=bool, default=False, help="write wav file", required=False)
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        if args.model_type == 'htdemucs' or args.model_type == 'hdemucs':
            config = OmegaConf.load(args.config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    print("Instruments: {}".format(config.training.instruments))

    model = get_model_from_config(args.model_type, config)

    print("Model loaded: {}".format(args.model_type))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.start_check_point != '':

        if args.model_type == 'dtt_net' and args.our_model is False:
            print('Load state dict')
            model.load_state_dict(torch.load(args.start_check_point)['state_dict'])
        else:
            from collections import OrderedDict
            state_dict = torch.load(args.start_check_point, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`  ## module 키 제거
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print(f"Loaded model")
        
        print('Start from checkpoint: {}'.format(args.start_check_point))

    device_ids = args.device_ids
    if type(device_ids)==int:
        device = torch.device(f'cuda:{device_ids}')
        model = model.to(device)
    else:
        device = torch.device(f'cuda:{device_ids[0]}')
        model = nn.DataParallel(model, device_ids=device_ids).to(device)

    valid(model, args, config, device, verbose=False)
    # evaluator = EVAL(config, args, model)
    # evaluator.run()


if __name__ == "__main__":
    check_validation(None)
