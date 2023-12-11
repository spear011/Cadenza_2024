# Hybrid Demucs with Self-Knowledge Distillation (PS-KD)

This repository introduces the Self-Knowledge Distillation with Progressive Refinement of Targets (PS-KD) approach to training Hybrid Demucs. The model training incorporates a curriculum learning approach. Additionally, it includes the codebase for the Cadenza Challenge 2024.

- Hybrid Demucs (V3) [[Paper]](https://arxiv.org/abs/2111.03600) [[Repo]](https://github.com/facebookresearch/demucs/tree/v3)
- PS-KD [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Self-Knowledge_Distillation_With_Progressive_Refinement_of_Targets_ICCV_2021_paper.html) [[Repo]](https://github.com/lgcnsai/PS-KD-Pytorch)

### Music Source Separation Training Code

Training code for the mss models based on [[here]](https://github.com/ZFTurbo/Music-Source-Separation-Training)

## Results

- The results indicate the performance of the validation set for the cadenza challenge.

|Title|PS-KD|Curriculum-learning|Augmentation|SDR avg|Haaqi Score|
|---|---|---|---|---|---|
|no fine-tune|-|-|-|3.701|0.6677|
|w/o aug|-|-|-|4.1838|0.6776|
|w/ aug|-|-|O|4.0762|0.6733|
|PS-KD w/o aug|O|-|-|4.2505|0.6764|
|PS-KD w/ aug|O|-|O|4.4060|0.6818|
|PS-KD Curri w/o aug|O|O|-|4.2002|0.6772|
|PS-KD Curri w/ aug|O|O|O|-|-|

### PS-KD
- The PS-KD method uses targets by creating soft targets, and we used an alpha of 0.8.

### Curriculum-learning
- We took the SDR score and used the top 75% of the scoring datasets as the EASY dataset.

### Augmentation
- We used a random augmentation method on pitch {-2, -1, 0, 1, 2} tempo {-20, -10, 0, 10, 20} with random augmentation. We also applied a random source mix, channel shuffle, and random audio effects (reverb, phaser, distortion) with a 0.05% probability. Soundstretch was used to augment pitch and tempo, and other techniques are available in code.