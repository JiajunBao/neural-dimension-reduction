
rm data/train.pth.tar
rm data/dev.pth.tar

TOP_K=20
LR=1e-3

python examples/train.py \
--input_dir data \
--output_dir checkpoints/1/top${TOP_K}/${LR} \
--learning_rate ${LR} \
--n_epoch 600 \
--per_gpu_batch_size 90000 \
--num_eval_per_epoch 2 \
--weight_decay 1e-5 \
--top_k ${TOP_K} \
--hidden_dims_list [200,100,50,25,20]

# rm data/train.pth.tar
# rm data/dev.pth.tar