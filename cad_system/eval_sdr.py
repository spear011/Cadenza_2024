import json
import logging
import warnings
from pathlib import Path

# pylint: disable=import-error
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
from mir_eval import separation
from tqdm import tqdm
import datetime as dt

from bandhdemucs.model.hdemucs import HDemucs

from enhance import decompose_signal
from evaluate import load_reference_stems
from clarity.utils.file_io import read_signal
from clarity.utils.audiogram import Listener
from clarity.utils.results_support import ResultsFile
from clarity.utils.source_separation_support import get_device, separate_sources
from recipes.cad_icassp_2024.baseline.evaluate import (
    apply_gains,
    apply_ha,
    make_scene_listener_list,
    remix_stems,
)

from bandhdemucs.utils.scores import compute_sdr, eval_sdr_score

logger = logging.getLogger(__name__)


@hydra.main(config_path="/Users/hai/Desktop/Cadenza/workspace/baseline/bandhdemucs", config_name="config")
def run(config: DictConfig) -> None:

    if config.separator.model == "demucs":

        separation_model = HDEMUCS_HIGH_MUSDB.get_model()
        model_sample_rate = HDEMUCS_HIGH_MUSDB.sample_rate
        sources_order = separation_model.sources
        normalise = True

    elif config.separator.model == "my_hdemucs":
        sources_order = ['drums', 'bass', 'other', 'vocals']
        check = '/Users/hai/Desktop/Cadenza/workspace/hdemucs.pth'
        separation_model = HDemucs(sources=sources_order)
        separation_model.load_state_dict(torch.load(check))
        model_sample_rate = 44100
        normalise = True

    else:
        raise ValueError(f"Unknown model {config.separator.model}")

    device, _ = get_device(config.separator.device)
    separation_model.to(device)

    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    with Path(config.path.music_file).open("r", encoding="utf-8") as file:
        songs = json.load(file)

        # Select a batch to process
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )

    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    previous_song = ""
    num_scenes = len(scene_listener_pairs)

    scores_headers = [
        "scene",
        "song",
        "listener",
        "vocals",
        "drums",
        "bass",
        "other",
        "overall",
    ]

    dates = dt.datetime.now().strftime("%m-%d")

    results_file = ResultsFile(
            f"sdr_{dates}_{config.separator.model}_{config.mode}.csv",
            header_columns=scores_headers,
        )

    overall_scores = np.array([])

    for idx, scene_listener_pair in enumerate(scene_listener_pairs, 1):

        scene_id, listener_id = scene_listener_pair

        scene = scenes[scene_id]
        song_name = f"{scene['music']}-{scene['head_loudspeaker_positions']}"

        logger.info(
            f"[{idx:03d}/{num_scenes:03d}] "
            f"Processing {scene_id}: {song_name} for listener {listener_id}"
        )
        # Get the listener's audiogram
        listener = listener_dict[listener_id]

        if song_name != previous_song:

            mixture_signal = read_signal(
                filename=Path(config.path.music_dir)
                / songs[song_name]["Path"]
                / "mixture.wav",
                sample_rate=config.sample_rate,
                allow_resample=True,
            )

            stems: dict[str, np.array] = decompose_signal(
                model=separation_model,
                model_sample_rate=model_sample_rate,
                signal=mixture_signal,
                signal_sample_rate=config.sample_rate,
                device=device,
                sources_list=sources_order,
                listener=listener,
                normalise=normalise,
            )

            reference_stems, _ = load_reference_stems(
                Path(config.path.music_dir) / songs[song_name]["Path"]
                )
            
            
            sdr_score = eval_sdr_score(
                reference_stems=reference_stems,
                output_stems=stems,
                sources_order=sources_order,
                sample_rate=config.sample_rate
            )

            logger.info(f"SDR score: {sdr_score['overall']:.2f}")

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
                }
            )

            overall_scores = np.append(overall_scores, sdr_score["overall"])

            if idx % 10 == 0 and idx != 0:
                logger.info(f"Overall score: {np.mean(overall_scores).round(3)}")
    
    logger.info(f"Overall score: {np.mean(overall_scores).round(3)}")
    logger.info('Done!')


if __name__ == "__main__":
    run()