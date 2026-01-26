import matplotlib.pyplot as plt
import argparse
import json
import os
from PIL import Image
from tqdm import tqdm
from gymnasium.envs.video_unshuffle.video_unshuffle import VideoUnshuffleEnv

def get_obs_label_gt_triple(env):
    obs, info = env.reset()
    obs = Image.fromarray(obs)
    label = info["label"]
    gt = info["correct_order"]
    return obs, label, gt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/video_unshuffle/static_data")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--sampling_strategy", type=str, default="salient")
    parser.add_argument("--min_frame_diff", type=float, default=1)
    parser.add_argument("--max_frames_to_analyze", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/video_unshuffle/static_data")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    env = VideoUnshuffleEnv(args.data_path, 
                            num_frames=args.num_frames, 
                            seed=args.seed, 
                            sampling_strategy=args.sampling_strategy, 
                            min_frame_diff=args.min_frame_diff, 
                            max_frames_to_analyze=args.max_frames_to_analyze)
    sample_idx = 0
    pbar = tqdm(total=args.num_samples, desc="Generating samples")
    while sample_idx < args.num_samples:
        try:
            obs, label, gt = get_obs_label_gt_triple(env)
            image_name = f"video_unshuffle_{sample_idx}.png"
            obs.save(os.path.join(args.output_dir, image_name))
            data = {
                "image_name": image_name,
                "label": label,
                "gt": gt
            }
            with open(os.path.join(args.output_dir, "ground_truth.jsonl"), "a") as f:
                f.write(json.dumps(data) + "\n")
            sample_idx += 1
            pbar.update(1)
        except Exception as e:
            print(f"Error on sample {sample_idx}: {e}")
            continue
    pbar.close()

if __name__ == "__main__":
    main()
