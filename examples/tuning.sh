set -x
#TOP_K=20
#LR=1e-4
#HIDDEN_DIMS=200-100-50-25-20

#for LR in 5e-7 4e-7 3e-7 2e-7 1e-7
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

#for LR in 6e-7 7e-7 8e-7 9e-7
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

#TOP_K=1
#LR=5e-7
#for HIDDEN_DIMS in 500-100-20 500-100-20-20 5000-1000-200-200
#do
#  python examples/train.py \
#  --input_dir data \
#  --output_dir checkpoints/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#  --learning_rate ${LR} \
#  --n_epoch 1200 \
#  --per_gpu_batch_size 90000 \
#  --num_eval_per_epoch 5 \
#  --weight_decay 1e-5 \
#  --top_k ${TOP_K} \
#  --hidden_dims_list ${HIDDEN_DIMS}
#done
#
#LR=5e-7
#HIDDEN_DIMS=200-100-50-25-20
#for TOP_K in {2..19}
#do
#  python examples/train.py \
#  --input_dir data \
#  --output_dir checkpoints/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#  --learning_rate ${LR} \
#  --n_epoch 1200 \
#  --per_gpu_batch_size 90000 \
#  --num_eval_per_epoch 2 \
#  --weight_decay 1e-5 \
#  --top_k ${TOP_K} \
#  --hidden_dims_list ${HIDDEN_DIMS}
#done
#
#
#TOP_K=20
#for HIDDEN_DIMS in 200-100-100-50-50-25-25-20 200-200-100-100-100-50-50-50-25-25-25-20-20 200-200-200-100-100-100-100-50-50-50-50-25-25-25-25-20-20-20 200-200-200-200-100-100-100-100-100-50-50-50-50-50-25-25-25-25-25-20-20-20-20
#do
#  for LR in 5e-7 5e-6 5e-5 5e-4 5e-3
#  do
#    python examples/train.py \
#    --input_dir data \
#    --output_dir checkpoints/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#    --learning_rate ${LR} \
#    --n_epoch 1200 \
#    --per_gpu_batch_size 90000 \
#    --num_eval_per_epoch 2 \
#    --weight_decay 1e-5 \
#    --top_k ${TOP_K} \
#    --hidden_dims_list ${HIDDEN_DIMS}
#  done
#done

#TOP_K=20
#for HIDDEN_DIMS in 200-100-100-50-50-25-25-20 200-200-100-100-100-50-50-50-25-25-25-20-20 200-200-200-100-100-100-100-50-50-50-50-25-25-25-25-20-20-20 200-200-200-200-100-100-100-100-100-50-50-50-50-50-25-25-25-25-25-20-20-20-20
#do
#  for LR in 5e-2 1e-2 5e-1 1e-1
#  do
#    python examples/train.py \
#    --input_dir data \
#    --output_dir checkpoints/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#    --learning_rate ${LR} \
#    --n_epoch 1200 \
#    --per_gpu_batch_size 90000 \
#    --num_eval_per_epoch 2 \
#    --weight_decay 1e-5 \
#    --top_k ${TOP_K} \
#    --hidden_dims_list ${HIDDEN_DIMS}
#  done
#done

TOP_K=1

for HIDDEN_DIMS in 500-100-20 500-100-20-20 5000-1000-200-200
do
  for LR in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1
  do
    python examples/train.py \
    --input_dir data \
    --output_dir checkpoints/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
    --learning_rate ${LR} \
    --n_epoch 1200 \
    --per_gpu_batch_size 90000 \
    --num_eval_per_epoch 5 \
    --weight_decay 1e-5 \
    --top_k ${TOP_K} \
    --hidden_dims_list ${HIDDEN_DIMS}
  done
done

# remove the old dataset

rm data/train.pth.tar
rm data/dev.pth.tar
TOP_K=1

for HIDDEN_DIMS in 500-100-20 500-100-20-20 5000-1000-200-200
do
  for LR in 1e-1 1e-2 1e-3 1e-4 1e-5  1e-6
  do
    python examples/train.py \
    --input_dir data \
    --output_dir checkpoints/random-anchors-${HIDDEN_DIMS}/top${TOP_K}/${LR} \
    --learning_rate ${LR} \
    --n_epoch 20 \
    --per_gpu_batch_size 90000 \
    --num_eval_per_epoch 5 \
    --weight_decay 0 \
    --top_k ${TOP_K} \
    --hidden_dims_list ${HIDDEN_DIMS}
  done
done

# rm data/train.pth.tar
# rm data/dev.pth.tar