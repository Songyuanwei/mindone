# Utility Scripts

This folder is a collection of utility scripts, listed and explained below.

> All scripts need to be run in the root path of project, unless otherwise noted.

## eval_videos_metrics.py

This script contains code and scripts for diffusion model evaluation, e.g.,

- CLIP Score For frame Consistency
- CLIP Score for Textual Alignment


Note that all the above metrics are computed based on neural network models.

> A convincing evaluation for diffusion models requires both visually qualitative comparision and quantitative measure. A higher CLIP score does not necessarily show one model is better than another.


#### CLIP Score for Frame Consistency

To compute the CLIP score on all frames of output video and report the average cosine similarity between all video frame pairs, please run

```shell
python ./scripts/eval_videos_metrics.py --video_data_dir <path-to-video-dir> --video_caption_path <path-to-video-caption-path> --model_name <HF-model-name>  --metric clip_score_frame
```

#### CLIP Score for Textual Alignment

To compute the average CLIP score between all frames of the output video and the corresponding editing prompts, please run

```shell
python ./scripts/eval_videos_metrics.py --video_data_dir <path-to-video-dir> --video_caption_path <path-to-video-caption-path> --model_name <HF-model-name>  --metric clip_score_text
```

Format of `.csv`:
```
video,caption
video_name1.mp4,"an airliner is taxiing on the tarmac at Dubai Airport"
video_name2.mp4,"a pigeon sitting on the street near the house"
...
```

## video_cut.py

`Video_cut.py` is a tool for processing videos, which can be used to generate video keyframes and cut video scenes.

#### Generate video keyframes

```
python .scripts/video_cut.py --video_data_path <path-to-videos> --save_dir <path-to-save-keyframes> --task keyframe
```
> Note: `keyframe_save_dir` defaults to None. If you do not set `keyframe_save_dir`, keyframes will be saved in the `video_data_path` folder

#### cut video scene

```
python .scripts/video_cut.py --video_data_path <path-to-videos> --task scene --save_dir <path-to-save-scene> --num_processing 1
```
> Note:

For more usage, please run `python .scripts/video_cut.py -h`.

## Reference

[1] https://github.com/showlab/loveu-tgve-2023/tree/main
