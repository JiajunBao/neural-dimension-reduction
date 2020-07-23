set -x
#TOP_K=20
#LR=1e-4
#HIDDEN_DIMS=200-100-50-25-20

#for LR in 5e-7 4e-7 3e-7 2e-7 1e-7
#do
#  python models/train.py \
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
#  python models/train.py \
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
#  python models/train.py \
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

#rm data/train.pth.tar
#rm data/dev.pth.tar
#
#LR=5e-7
#HIDDEN_DIMS=200-100-50-25-20
#for TOP_K in 1 3 5 7 9 11 13 15 17 19 21
#do
#  python models/train.py \
#  --input_dir data \
#  --output_dir checkpoints/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#  --learning_rate ${LR} \
#  --n_epoch 600 \
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
#    python models/train.py \
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
#    python models/train.py \
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


# remove the old dataset
# a random anchor
#rm data/train.pth.tar
#rm data/dev.pth.tar
#TOP_K=1
#
#for HIDDEN_DIMS in 500-100-20 500-100-20-20 5000-1000-200-200
#do
#  for LR in 5e-7 1e-2 1e-3 1e-4 1e-5  1e-6
#  do
#    rm -rf checkpoints/random-anchors-${HIDDEN_DIMS}/top${TOP_K}/${LR}
#    python models/train.py \
#    --input_dir data \
#    --output_dir checkpoints/random-anchors-${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#    --learning_rate ${LR} \
#    --n_epoch 600 \
#    --per_gpu_batch_size 90000 \
#    --num_eval_per_epoch 5 \
#    --weight_decay 0 \
#    --top_k ${TOP_K} \
#    --hidden_dims_list ${HIDDEN_DIMS}
#  done
#done
#

#rm data/train.pth.tar
#rm data/dev.pth.tar
#
#TOP_K=1
#
#for HIDDEN_DIMS in 500-100-20 500-100-20-20 5000-1000-200-200
#do
#  for LR in 5e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1
#  do
#    python models/train.py \
#    --input_dir data \
#    --output_dir checkpoints/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#    --learning_rate ${LR} \
#    --n_epoch 600 \
#    --per_gpu_batch_size 90000 \
#    --num_eval_per_epoch 5 \
#    --weight_decay 1e-5 \
#    --top_k ${TOP_K} \
#    --hidden_dims_list ${HIDDEN_DIMS}
#  done
#done

#rm data/train.pth.tar
#rm data/dev.pth.tar
#
#TOP_K=20
#LR=5e-7
#for HIDDEN_DIMS in 5000-100-20-20 5000-1000-20-20 5000-1000-200-20
#do
#  python models/train.py \
#  --input_dir data \
#  --output_dir checkpoints/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#  --learning_rate ${LR} \
#  --n_epoch 600 \
#  --per_gpu_batch_size 90000 \
#  --num_eval_per_epoch 5 \
#  --weight_decay 1e-5 \
#  --top_k ${TOP_K} \
#  --hidden_dims_list ${HIDDEN_DIMS}
#done

# rm data/train.pth.tar
# rm data/dev.pth.tar

#TOP_K=20
#LR=1e-4
#HIDDEN_DIMS=200-100-50-25-20
#
#for LR in 1e-7 1e-6 1e-5 1e-4 1e-3
#do
#  python models/train.py \
#  --input_dir data \
#  --output_dir checkpoints/normalized-output-${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#  --learning_rate ${LR} \
#  --n_epoch 1200 \
#  --per_gpu_batch_size 90000 \
#  --num_eval_per_epoch 2 \
#  --weight_decay 1e-5 \
#  --top_k ${TOP_K} \
#  --hidden_dims_list ${HIDDEN_DIMS}
#done

#TOP_K=1000
##LR=1e-4
#HIDDEN_DIMS=200-100-50-25-20
#
#
#for LR in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2
#do
#  python models/train.py \
#  --input_dir data/processed/sample \
#  --output_dir checkpoints/sample/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#  --learning_rate ${LR} \
#  --n_epoch 1200 \
#  --per_gpu_batch_size 90000 \
#  --num_eval_per_epoch 2 \
#  --weight_decay 1e-5 \
#  --top_k ${TOP_K} \
#  --hidden_dims_list ${HIDDEN_DIMS}
#done

#TOP_K=2000
#
#
#for LR in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2
#do
#  for HIDDEN_DIMS in 5000-1000-200-200-200 5000-2000-300-300-300 5000-3000-400-400-400
#  do
#    python models/train.py \
#    --input_dir data/processed/sample \
#    --output_dir checkpoints/sample/${HIDDEN_DIMS}/top${TOP_K}/${LR} \
#    --learning_rate ${LR} \
#    --n_epoch 1200 \
#    --per_gpu_batch_size 90000 \
#    --num_eval_per_epoch 2 \
#    --weight_decay 1e-5 \
#    --top_k ${TOP_K} \
#    --hidden_dims_list ${HIDDEN_DIMS}
#  done
#done
#
#
#
#LR=1e-7
#HIDDEN_DIMS=5000-3000-400-400-400
#
#for TOP_K in 1998 1996 1994 1992 1990 1988 1986 1984 1982 1980 1978 1976 1974 1972 1970 1968
#do
#  python models/train.py \
#  --input_dir data/processed/sample \
#  --output_dir checkpoints/sample/"${HIDDEN_DIMS}"/top${TOP_K}/${LR} \
#  --learning_rate ${LR} \
#  --n_epoch 1200 \
#  --per_gpu_batch_size 90000 \
#  --num_eval_per_epoch 2 \
#  --weight_decay 1e-5 \
#  --top_k ${TOP_K} \
#  --hidden_dims_list "${HIDDEN_DIMS}"
#done
#

TOP_K=2000


for LR in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2
do
  for HIDDEN_DIMS in 5000-1000-200-200-200 5000-2000-300-300-300 5000-3000-400-400-400
  do
    python models/train.py \
    --input_dir data/processed/sample \
    --output_dir checkpoints/shortcut-sample/"${HIDDEN_DIMS}"/top${TOP_K}/${LR} \
    --learning_rate ${LR} \
    --n_epoch 1200 \
    --per_gpu_batch_size 90000 \
    --num_eval_per_epoch 2 \
    --weight_decay 1e-5 \
    --top_k ${TOP_K} \
    --hidden_dims_list "${HIDDEN_DIMS}" \
    --add_shortcut True
  done
done


LR=1e-7
HIDDEN_DIMS=5000-3000-400-400-400

for TOP_K in 1998 1996 1994 1992 1990 1988 1986 1984 1982 1980 1978 1976 1974 1972 1970 1968
do
  python models/train.py \
  --input_dir data/processed/sample \
  --output_dir checkpoints/shortcut-sample/"${HIDDEN_DIMS}"/top${TOP_K}/${LR} \
  --learning_rate ${LR} \
  --n_epoch 1200 \
  --per_gpu_batch_size 90000 \
  --num_eval_per_epoch 2 \
  --weight_decay 1e-5 \
  --top_k ${TOP_K} \
  --hidden_dims_list "${HIDDEN_DIMS}" \
  --add_shortcut True
done


TOP_K=2000
LR=5e-7

for HIDDEN_DIMS in 200-100-100-50-50-25-25-20 200-200-100-100-100-50-50-50-25-25-25-20-20 200-200-200-100-100-100-100-50-50-50-50-25-25-25-25-20-20-20 200-200-200-200-100-100-100-100-100-50-50-50-50-50-25-25-25-25-25-20-20-20-20
do
  python models/train.py \
  --input_dir data/processed/sample \
  --output_dir checkpoints/shortcut-sample/"${HIDDEN_DIMS}"/top${TOP_K}/${LR} \
  --learning_rate ${LR} \
  --n_epoch 1200 \
  --per_gpu_batch_size 90000 \
  --num_eval_per_epoch 2 \
  --weight_decay 1e-5 \
  --top_k ${TOP_K} \
  --hidden_dims_list "${HIDDEN_DIMS}" \
  --add_shortcut True
done
