# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import numpy as np
import torch
import torch.nn as nn

from clarity.utils.signal_processing import (
    clip_signal,
    denormalize_signals,
    normalize_signal,
    resample,
    to_16bit,
)
from clarity.utils.source_separation_support import get_device, separate_sources
from numpy import ndarray
from clarity.utils.audiogram import Listener


def decompose_signal(
    model: torch.nn.Module,
    model_sample_rate: int,
    signal: ndarray,
    signal_sample_rate: int,
    device: torch.device,
    sources_list: list,
    listener: Listener,
    normalise: bool = True,
) -> dict:
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
        signal,
        model_sample_rate,
        device=device,
    )
    # # only one element in the batch
    sources = sources[0]
    ref = ref.cpu().numpy()

    if normalise:
        sources = denormalize_signals(sources, ref)

    sources = np.transpose(sources, (0, 2, 1))
    return dict(zip(sources_list, sources))



def get_model_from_config(model_type, config):
    if model_type == 'mdx23c':
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'mel_band_roformer':
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(
            **dict(config.model)
        )
    elif model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(
            **dict(config.model)
        )
    elif model_type == 'hdemucs':
        # from models.demucs.hdemucs import HDemucs, get_model
        # from models.demucs.demucs import Demucs
        # model = get_model(config)
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
        bundle = HDEMUCS_HIGH_MUSDB
        model = bundle.get_model()
    
    elif model_type == 'bs_hdemucs':
        from models.demucs.band_hdemucs import BandHDemucs
        from models.demucs4ht import get_model
        model = get_model(config)

    elif model_type == 'dtt_net':
        from models.dp_tdf.dp_tdf_net import DPTDFNet
        model = DPTDFNet(
            **dict(config.model)
        )
    else:
        print('Unknown model: {}'.format(model_type))
        model = None

    return model

def demix_track(config, model, mix, device):
    C = config.audio.hop_length * (config.inference.dim_t - 1)
    N = config.inference.num_overlap
    step = C // N

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)
            i = 0
            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                x = model(part.unsqueeze(0))[0]
                result[..., i:i+length] += x[..., :length]
                counter[..., i:i+length] += 1.
                i += step

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if config.training.target_instrument is None:
        return {k: v.T for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return {k: v.T for k, v in zip([config.training.target_instrument], estimated_sources)}


def demix_track_demucs(config, model, mix, device):
    S = len(config.training.instruments)
    C = config.training.samplerate * config.training.segment
    N = config.inference.num_overlap
    step = C // N
    # print(S, C, N, step, mix.shape, mix.device)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            mix = mix.to(device)
            req_shape = (S, ) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)
            i = 0
            all_parts = []
            all_lengths = []
            all_steps = []
            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                all_parts.append(part)
                all_lengths.append(length)
                all_steps.append(i)
                i += step
            all_parts = torch.stack(all_parts, dim=0)
            # print(all_parts.shape)

            start_time = time.time()
            res = model(all_parts)
            # print(res.shape)
            # print("Time:", time.time() - start_time)
            # print(part.mean(), part.max(), part.min())
            # print(x.mean(), x.max(), x.min())

            for j in range(res.shape[0]):
                x = res[j]
                length = all_lengths[j]
                i = all_steps[j]
                # Sometimes model gives nan...
                if torch.isnan(x[..., :length]).any():
                    result[..., i:i+length] += all_parts[j][..., :length].to(device)
                else:
                    result[..., i:i + length] += x[..., :length]
                counter[..., i:i+length] += 1.

            # print(result.mean(), result.max(), result.min())
            # print(counter.mean(), counter.max(), counter.min())
            estimated_sources = result / counter

    if S > 1:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources.cpu().numpy())}
    else:
        return estimated_sources
    

def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)


def compute_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), dim=(2, 3))
    den = torch.sum(torch.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores

def eval_sdr_score(
        reference_stems,
        output_stems,
        sources_order,
        sample_rate):
    
    # for inst in sources_order:
    #     if inst not in output_stems:
    #         raise ValueError(f"Missing {inst} in output stems")
    #     if inst not in reference_stems:
    #         raise ValueError(f"Missing {inst} in reference stems")

    # ref_stack = np.stack([reference_stems[inst] for inst in sources_order])
    # est_stack = np.stack([output_stems[inst] for inst in sources_order])

    ref_stack = torch.from_numpy(reference_stems).unsqueeze(0)
    est_stack = torch.from_numpy(output_stems).unsqueeze(0)

    sdr_scores = dict()

    scores = compute_sdr(ref_stack, est_stack).squeeze(0).numpy()

    for idx, inst in enumerate(sources_order):
        sdr_scores[inst] = scores[idx]

    sdr_scores['overall'] = np.mean(list(sdr_scores.values()))

    return sdr_scores
