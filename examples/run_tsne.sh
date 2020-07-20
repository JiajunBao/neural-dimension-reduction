
python src/models/tSNE/tSNE.py \
--input_path data/processed/sample/test.csv \
--output_dir checkpoints/tsne/sample-test-set \
--n_iter 4000


python src/models/tSNE/tSNE.py \
--input_path data/processed/sample/train.csv \
--output_dir checkpoints/tsne/sample-train-set \
--n_iter 4000
