import matplotlib.pyplot as plt
import argparse
import json
import os
from PIL import Image
from tqdm import tqdm
from gymnasium.envs.zoom_in.zoom_in import ZoomInEnv

def get_obs_gt_pair(env):
    obs, info = env.reset()
    obs = Image.fromarray(obs)
    gt = info["original_view_order"]
    return obs, gt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default="data/zoom_in/samples")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_zoom_level", type=float, default=1.5)
    parser.add_argument("--num_zoom_views", type=int, default=4)
    parser.add_argument("--zoom_gap", type=float, default=1.0)
    parser.add_argument("--zoom_std", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="data/zoom_in/static_data")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ground_truth_jsonl = []
    env = ZoomInEnv(sample_dir=args.sample_dir, 
                    seed=args.seed, 
                    min_zoom_level=args.min_zoom_level, 
                    num_zoom_views=args.num_zoom_views, 
                    zoom_gap=args.zoom_gap, 
                    zoom_std=args.zoom_std)
    
    for i in tqdm(range(args.num_samples)):
        image_name = f"zoom_in_{i}.png"
        obs, gt = get_obs_gt_pair(env)
        ground_truth_jsonl.append({"image_name": image_name, "gt": gt})
        obs.save(os.path.join(args.output_dir, image_name))

    with open(os.path.join(args.output_dir, "ground_truth.jsonl"), "w") as f:
        for item in ground_truth_jsonl:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()