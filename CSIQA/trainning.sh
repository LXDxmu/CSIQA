### LIVE_test
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 --master_port 10538 main.py \
--cfg configs/Pure/vit_small_pre_coder_live.yaml \
--data-path /home/data/hrz/data/qgy/IQA-Dataset/live/databaserelease2  \
--output live_log \
--tensorboard \
--tag 0519 \
--repeat \
--rnum 2
