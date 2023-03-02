# Training for FFHQ



## Train network

Run the following command:
```
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=8 --nproc_per_node=1 train_128_2_256_dist.py --name DATASET_NAME --batch_size 8
```