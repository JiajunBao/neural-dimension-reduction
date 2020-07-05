

python examples/train.py \
--input_dir data \
--output_dir checkpoints/200-100-50-25-20/1e-3 \
--learning_rate 1e-3 \
--n_epoch 600 \
--per_gpu_batch_size 90000 \
--num_eval_per_epoch 2 \
--weight_decay 1e-5

# rm data/train.pth.tar
# rm data/dev.pth.tar