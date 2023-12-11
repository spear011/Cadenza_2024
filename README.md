# Hybrid Demucs with Self-Knowledge Distillation (PS-KD)

This repository introduces the Self-Knowledge Distillation with Progressive Refinement of Targets (PS-KD) approach to training Hybrid Demucs. and also the code for the cadenza challenge 2024.

- Hybrid Demucs (V3) [[Paper]](https://arxiv.org/abs/2111.03600) [[Repo]](https://github.com/facebookresearch/demucs/tree/v3)
- PS-KD [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Self-Knowledge_Distillation_With_Progressive_Refinement_of_Targets_ICCV_2021_paper.html) [[Repo]](https://github.com/lgcnsai/PS-KD-Pytorch)

### Music Source Separation Training Code

Training code for the mss models based on [[here]](https://github.com/ZFTurbo/Music-Source-Separation-Training)


# Getting Started

## Setup
1. Clone this repository
2. Download Datasets (ComMU, NSynth)
3. Install required packages
    ```
    pip install -r requirements.txt
    ```

## Preparation
- The ComMU dataset can be preprocessed to achieve balanced Track role classes.
    ```
    $ python preparation.py data-folder
    ```
- After successful preprocessing, project tree would be like this,
  ```
    .
    ├── commu_meta.csv
    ├── commu_midi
    └── balanced
        ├── balanced_meta.csv
        ├── train
        │   ├── raw
        ├── valid
        │   ├── raw
        └── test
            └── raw

    ```

## Augmentation
- You can augment training data provided by the [[ComMU-code]](https://github.com/POZAlabs/ComMU-code). The augmentation process will only involve the training data.
    ```
    $ python preprocess.py --root_dir ./data-folder/balanced --csv_path ./data-folder/balanced/balanced_meta.csv
    ```

## MIDI to Audio
    ```
    $ python synthesize.py Nsynth-dir data-folder output-dir
    ```

# Audio Output Results

- Output audio file name is same as the MIDI file with the preset and source information added. 
  - 'midifile-name_preset_source.wav' 
- The files in the /output folder looks like this

```bash
output_dir/audio/train/midifile01_030_1.wav
output_dir/audio/train/midifile02_002_0.wav
output_dir/audio/val/midifile03_random_random.wav
```

- Output csv: synthesized_results.csv

|id|instrument|preset|source|
|---|---|---|---|
|midifile01|keyboard|030|1|
|midifile02|guitar|002|0|
|midifile03|guitar|random|random|
