source /home/zw1300/anaconda3/etc/profile.d/conda.sh
conda activate vr

python video_unshuffle.py \
    --data_path /scratch/gpfs/DANQIC/zw1300/data/SS2 \
    --num_samples 30000 \
    --seed 0 \
    --num_frames 4 \
    --sampling_strategy salient \
    --min_frame_diff 1 \
    --max_frames_to_analyze 100 \
    --output_dir /scratch/gpfs/DANQIC/zw1300/data/VSP/video_unshuffle