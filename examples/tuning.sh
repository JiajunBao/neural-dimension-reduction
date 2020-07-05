TOP_K=20
#LR=1e-4
HIDDEN_DIMS=200-100-50-25-20

for LR in 1e-6 5e-5 1e-5 5e-4 1e-4 5e-3 1e-3
do
  python examples/train.py \
  --input_dir data \
  --output_dir checkpoints/1/top${TOP_K}/${LR} \
  --learning_rate ${LR} \
  --n_epoch 1200 \
  --per_gpu_batch_size 90000 \
  --num_eval_per_epoch 2 \
  --weight_decay 1e-5 \
  --top_k ${TOP_K} \
  --hidden_dims_list ${HIDDEN_DIMS}
done


#for TOP_K in {1..20}
#do
#  python examples/train.py \
#  --input_dir data \
#  --output_dir checkpoints/1/top${TOP_K}/${LR} \
#  --learning_rate ${LR} \
#  --n_epoch 1200 \
#  --per_gpu_batch_size 90000 \
#  --num_eval_per_epoch 2 \
#  --weight_decay 1e-5 \
#  --top_k ${TOP_K} \
#  --hidden_dims_list ${HIDDEN_DIMS}
#done



# rm data/train.pth.tar
# rm data/dev.pth.tar