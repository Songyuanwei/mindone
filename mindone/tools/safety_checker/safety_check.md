# Safety Checker

## Introduction
To validate the safeness of images or videos generated by a model, we implement two safety checkers in `safety_checker.py` with `safety_version=1` and `safety_version=2`(Default).

`safety_version=1` applies CLIP to compute the similarity between the given images and a pre-defined list of NSFW concepts (two possible `.yaml` files to choose from). This is consistent with the one used in CompVis's stable diffusion 1.x from `diffusers` implementation. The output is a list of True or False with bad concepts.

`safety_version=2` trains a NSFW classifier with a supervised approach, taking image features from CLIP as its input. This is consistent with the one used in StabilityAI's stable diffusion 2.x. The output is a number in 0-1, representing the probability of the generated image being NSFW.

## Pre-requisites
You need to download the checkpoint file for a CLIP model of your choice. Download links for some models are provided below.

- [clip_vit_b_16](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/clip/clip_vit_b_16.ckpt)
- [clip_vit_b_32](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/clip_vit_b_32.ckpt)
- [clip_vit_l_14](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/clip/clip_vit_l_14.ckpt) (Default)
> Note: when `safety_version=2`, safety checker currently only supports `clip_vit_l_14`.

For other compatible models, e.g., OpenCLIP, you can download `open_clip_pytorch_model.bin` from HuggingFace (HF) and then convert to `.ckpt` using `tools/_common/clip/convert_weight.py`. When using a model other than the default, you should supply the path to your model's config file. Some useful examples are provided in `tools/_common/clip/configs`.


## Usage
You can incorporate the checkers into your image generation pipelines. By default, they are included in `text_to_image.py` for evaluating the safety of stable diffusion outputs. Additionally, you have the option to test the checkers on saved images or videos. Please execute the following command:

### Image Safety Check

- MindSpore backend
```
export PYTHONPATH="your-mindone-path"
python mindone/tools/safety_checker/safety_checker.py --ckpt_path <path-to-model> --image_path_or_dir <path-to-image>
```
- PyTorch backend
```
export PYTHONPATH="your-mindone-path"
python tools/safety_checker/safety_checker.py --backend pt --image_path_or_dir <path-to-image>
```
> Note: If you want to no-check-certificate, please set `os.environ["CURL_CA_BUNDLE"] = ""` in `safety_checker.py`

By default, we use MindSpore backend for CLIP score computing. You may swich to use `torch` and `transformers` by setting `--backend=pt`.

For more usage, please run `python tools/safety_checker/safety_checker.py -h`.

`image_path_or_dir` should lead to an image file or a directory containing images. If it is a directory, then the images are sorted by their filename in an ascending order.

### Video Safety Check

In video safety checks, keyframes are extracted from the video and then subjected to potential NSFW content detection.

- MindSpore backend
```
export PYTHONPATH="your-mindone-path"
python mindone/tools/safety_checker/safety_checker.py --ckpt_path <path-to-model> --video_path_or_dir <path-to-image>
```
- PyTorch backend
```
export PYTHONPATH="your-mindone-path"
python tools/safety_checker/safety_checker.py --backend pt --video_path_or_dir <path-to-image>
```
> Note: If you want to no-check-certificate, please set `os.environ["CURL_CA_BUNDLE"] = ""` in `safety_checker.py`

By default, we use MindSpore backend for CLIP score computing. You may swich to use `torch` and `transformers` by setting `--backend=pt`.

For more usage, please run `python tools/safety_checker/safety_checker.py -h`.

`video_path_or_dir` should lead to an video file or a directory containing videos. If it is a directory, then the videos are sorted by their filename in an ascending order.


## Reference

[1] https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py
[2] https://github.com/LAION-AI/CLIP-based-NSFW-Detector