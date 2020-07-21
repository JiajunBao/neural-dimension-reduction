
python src/models/tSNE/tSNE.py \
--input_path data/processed/sample/dev.csv \
--output_dir checkpoints/tsne/sample-dev-set \
--dim_out 20 \
--n_iter 4000


python src/models/tSNE/tSNE.py \
--input_path data/processed/sample/train.csv \
--output_dir checkpoints/tsne/sample-train-set \
--dim_out 20 \
--n_iter 4000
