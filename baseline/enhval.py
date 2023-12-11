""" enhance and evaluate. """
from __future__ import annotations

import json
import logging
from pathlib import Path

# pylint: disable=import-error
import hydra
import numpy as np
import torch
from numpy import ndarray
from omegaconf import DictConfig
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

from evaluate import load_reference_stems
from bandhdemucs.utils.scores import eval_sdr_score
from clarity.utils.results_support import ResultsFile
import datetime as dt
from tqdm import tqdm

import pyloudnorm as pyln
from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal
from clarity.utils.flac_encoder import FlacEncoder
from clarity.utils.signal_processing import (
    clip_signal,
    denormalize_signals,
    normalize_signal,
    resample,
    to_16bit,
)
from clarity.utils.source_separation_support import get_device, separate_sources
from recipes.cad_icassp_2024.baseline.evaluate import (
    apply_gains,
    apply_ha,
    make_scene_listener_list,
    remix_stems,
)
from clarity.utils.signal_processing import compute_rms, resample
from clarity.evaluator.haaqi import compute_haaqi
from utils import demix_track, demix_track_demucs, sdr, get_model_from_config, eval_sdr_score, decompose_signal
from enhance import process_remix_for_listener


class EVAL(object):
    def __init__(self,
                 config,
                 args,
                 separation_model):
        self.config = config
        self.args = args
        self.model = separation_model
        self.listener_dict, self.gains, self.scenes, self.scene_listener_pairs, self.songs = self.load_meta()
        self.enhancer = NALR(**config.nalr)
        self.compressor = Compressor(**config.compressor)
        self.scores_headers = ["scene", "song", "listener", "vocals", "drums", "bass", "other", "overall", "left_score", "right_score", "score"]

    def load_meta(self):
        """Load metadata from config."""
        config = self.config

        listener_dict = Listener.load_listener_dict(config.path.listeners_file)

        with Path(config.path.gains_file).open("r", encoding="utf-8") as file:
            gains = json.load(file)

        with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
            scenes = json.load(file)

        with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
            scenes_listeners = json.load(file)

        with Path(config.path.music_file).open("r", encoding="utf-8") as file:
            songs = json.load(file)
        
        scene_listener_pairs = make_scene_listener_list(scenes_listeners, config.evaluate.small_test)
        scene_listener_pairs = scene_listener_pairs[config.evaluate.batch :: config.evaluate.batch_size]
        return listener_dict, gains, scenes, scene_listener_pairs, songs
    
    def run(self, epoch):

        dates = dt.datetime.now().strftime("%m-%d")

        config = self.config
        args = self.args
        enhancer = self.enhancer
        compressor = self.compressor
        model = self.model
        listener_dict = self.listener_dict
        gains = self.gains
        scenes = self.scenes
        scene_listener_pairs = self.scene_listener_pairs
        songs = self.songs
        scores_headers = self.scores_headers

        results_file = ResultsFile(
            f"sdr_{dates}_{config.separator.model}_{epoch}.csv",
            header_columns=scores_headers,
        )
        
        instruments = config.training.instruments
        if config.training.target_instrument is not None:
            instruments = [config.training.target_instrument]

        all_sdr = dict()
        for instr in config.training.instruments:
            all_sdr[instr] = []

        overall_scores = np.array([])
        num_scenes = len(scene_listener_pairs)

        sr = config.sample_rate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()

        for idx, scene_listener_pair in tqdm(enumerate(scene_listener_pairs, 1)):
            scene_id, listener_id = scene_listener_pair

            scene = scenes[scene_id]
            song_name = f"{scene['music']}-{scene['head_loudspeaker_positions']}"

            # Get the listener's audiogram
            listener = listener_dict[listener_id]

            mix = read_signal(
                filename=Path(config.path.music_dir)
                / songs[song_name]["Path"]
                / "mixture.wav",
                sample_rate=config.sample_rate,
                allow_resample=True,
            )

            if args.model_type == 'htdemucs' or args.model_type == 'hdemucs':

                stems: dict[str, np.array] = decompose_signal(
                                            model=model,
                                            model_sample_rate=sr,
                                            signal=mix,
                                            signal_sample_rate=sr,
                                            device=device,
                                            sources_list=instruments,
                                            listener=None,
                                            normalise=True,
                                            )
            elif args.model_type == 'dtt_net':
                mixture = torch.tensor(mix, dtype=torch.float32)
                res = demix_track(config, model, mixture, device)
            else:
                res = demix_track(config, model, mixture, device)


            if 1:  
                pbar_dict = {}

                reference_stems, _ = load_reference_stems(
                    Path(config.path.music_dir) / songs[song_name]["Path"]
                    )

                sdr_score = eval_sdr_score(
                            reference_stems=reference_stems,
                            output_stems=stems,
                            sources_order=instruments,
                            sample_rate=config.sample_rate
                            )
                
                overall_scores = np.append(overall_scores, sdr_score["overall"])
                stems = apply_gains(stems, config.sample_rate, gains[scene["gain"]])
                enhanced_signal = remix_stems(stems, mix, sr)

                enhanced_signal = process_remix_for_listener(
                        signal=enhanced_signal,
                        enhancer=enhancer,
                        compressor=compressor,
                        listener=listener,
                        apply_compressor=config.apply_compressor,
                    )

                reference_stems = apply_gains(reference_stems, config.sample_rate, gains[scene["gain"]])
                reference_mixture = remix_stems(reference_stems, mix, config.sample_rate)

                # Apply hearing aid to reference signals
                left_reference = apply_ha(
                    enhancer=enhancer,
                    compressor=None,
                    signal=reference_mixture[:, 0],
                    audiogram=listener.audiogram_left,
                    apply_compressor=False,
                )
                right_reference = apply_ha(
                    enhancer=enhancer,
                    compressor=None,
                    signal=reference_mixture[:, 1],
                    audiogram=listener.audiogram_right,
                    apply_compressor=False,
                )

                # Compute the scores
                left_score = compute_haaqi(
                    processed_signal=resample(
                        enhanced_signal[:, 0],
                        config.remix_sample_rate,
                        config.HAAQI_sample_rate,
                    ),
                    reference_signal=resample(
                        left_reference, config.sample_rate, config.HAAQI_sample_rate
                    ),
                    processed_sample_rate=config.HAAQI_sample_rate,
                    reference_sample_rate=config.HAAQI_sample_rate,
                    audiogram=listener.audiogram_left,
                    equalisation=2,
                    level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 0])),
                )

                right_score = compute_haaqi(
                    processed_signal=resample(
                        enhanced_signal[:, 1],
                        config.remix_sample_rate,
                        config.HAAQI_sample_rate,
                    ),
                    reference_signal=resample(
                        right_reference, config.sample_rate, config.HAAQI_sample_rate
                    ),
                    processed_sample_rate=config.HAAQI_sample_rate,
                    reference_sample_rate=config.HAAQI_sample_rate,
                    audiogram=listener.audiogram_right,
                    equalisation=2,
                    level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 1])),
                )

                song_score = float(np.mean([left_score, right_score]))

                results_file.add_result(
                    {
                    "scene": scene_id,
                    "song": song_name,
                    "listener": listener.id,
                    "vocals": sdr_score["vocals"],
                    "drums": sdr_score["drums"],
                    "bass": sdr_score["bass"],
                    "other": sdr_score["other"],
                    "overall": sdr_score["overall"],
                    "left_score": left_score,
                    "right_score": right_score,
                    "score": song_score,
                    }
                )

                pbar_dict['song_score'] = song_score

        return song_score





            