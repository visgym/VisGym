source /home/zw1300/anaconda3/etc/profile.d/conda.sh
conda activate vr

python zoom_in.py \
    --sample_dir /scratch/gpfs/DANQIC/zw1300/data/llava_pretrain_images \
    --num_samples 20000 \
    --output_dir /scratch/gpfs/DANQIC/zw1300/data/VSP/zoom_in \
    --seed 42 \
    --min_zoom_level 1.5 \
    --num_zoom_views 4 \
    --zoom_gap 1.0 \
    --zoom_std 0.2
