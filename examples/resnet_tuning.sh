#TOP_K=2000
#LR=1e-3
#
#for HIDDEN_DIMS in resnet32 resnet47
#do
#  python examples/train_resnet.py \
#  --input_dir data/processed/sample \
#  --output_dir checkpoints/sample/"${HIDDEN_DIMS}"/top${TOP_K}/${LR} \
#  --learning_rate ${LR} \
#  --n_epoch 2000 \
#  --per_gpu_batch_size 90000 \
#  --num_eval_per_epoch 2 \
#  --weight_decay 1e-6 \
#  --top_k ${TOP_K} \
#  --hidden_dims_list "${HIDDEN_DIMS}" \
#  --add_shortcut True
#done

TOP_K=2000
HIDDEN_DIMS=resnet47

for LR in 1e-5 2e-5 4e-5 8e-5 1.6e-4 3.2e-4 6.4e-4
do
  python examples/train_resnet.py \
  --input_dir data/processed/sample \
  --output_dir checkpoints/sample/"${HIDDEN_DIMS}"/top${TOP_K}/${LR} \
  --learning_rate ${LR} \
  --n_epoch 2000 \
  --per_gpu_batch_size 90000 \
  --num_eval_per_epoch 2 \
  --weight_decay 1e-6 \
  --top_k ${TOP_K} \
  --hidden_dims_list "${HIDDEN_DIMS}" \
  --add_shortcut True
done