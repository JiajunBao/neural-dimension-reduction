LR=1e-3
MODEL_NAME=resnet32
TOP_K=2000

rm data/processed/sample/train.insane.pth.tar
rm data/processed/sample/dev.insane.pth.tar

python examples/train_resnet.py \
--input_dir data/processed/sample \
--output_dir checkpoints/sample/"${MODEL_NAME}"/top${TOP_K}/${LR} \
--learning_rate ${LR} \
--n_epoch 1200 \
--per_gpu_batch_size 90000 \
--num_eval_per_epoch 2 \
--weight_decay 1e-6 \
--top_k ${TOP_K} \
--config_name "${MODEL_NAME}"