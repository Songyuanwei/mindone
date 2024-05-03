import argparse

from mindone.data import KeyFrameExtractUtils, SceneCutUtils

parser = argparse.ArgumentParser()
parser.add_argument("--video_data_path", type=str, default=None, help="path to video data." "Default: None")
parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="path for save keyframe images or sences. if None, keyframes or sences will save in video_data_path, ."
    "Default: None",
)
parser.add_argument(
    "--task",
    type=str,
    default="keyframe",
    choices=["keyframe", "scene"],
    help="Choose how to process videos." "Default: keframe",
)
parser.add_argument(
    "--Detectortype",
    type=str,
    default="AdaptiveDetector",
    choices=["ContentDetector", "AdaptiveDetector", "ThresholdDetector"],
    help="scene detection algorithms type." "Default: AdaptiveDetector",
)
parser.add_argument(
    "--num_processing",
    type=int,
    default=1,
    help="multiple processesr",
)
args = parser.parse_args()

if args.task == "keyframe":
    keyframe = KeyFrameExtractUtils(args.video_data_path, args.save_dir)
    keyframe.save_keyframe()
elif args.task == "scene":
    cut_scene = SceneCutUtils(args.video_data_path, args.Detectortype, args.save_dir, args.num_processing)
    cut_scene.scenes_detect()
else:
    raise ValueError(f"Unsupported tasks: {args.task}.")
