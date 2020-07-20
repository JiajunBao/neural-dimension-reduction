
python src/models/tSNE/tSNE.py \
--input_path data/processed/sample/dev.csv \
--output_dir checkpoints/tsne/sample-test-set \
--dim_out 2 \
--n_iter 4000


python src/models/tSNE/tSNE.py \
--input_path data/processed/sample/train.csv \
--output_dir checkpoints/tsne/sample-train-set \
--n_iter 4000
