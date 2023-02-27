#! /bin/bash
#MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 1 --num_heads 4 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --num_head_channels 64"
#DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm True --resblock_updown True"
#TRAIN_FLAGS="--lr 2e-5 --batch_size 4"
path=$0
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=2 train_dist1.py --real_data_dir $1