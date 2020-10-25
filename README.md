Neural-based Dimension Reduction
==============================

Neural-based Dimension Reduction Techniques for Nearest Neighbor Preservation.

Set up the environment
------------
```
# create a new environment
python3 -m venv env

# activate the environment
source env/bin/activate

# install packages
pip instal -r requirments.txt
```

kl diveragence-based model
------------
1. The original code from the thesis is at `./reference`.
2. A better implementation is at `./src/models`. Shell code for running the code is `./examples`.
```
# tsne baselines
python src/models/tSNE/tSNE.py \
--input_path data/processed/sample/train.csv \
--output_dir checkpoints/tsne/sample-train-set \
--dim_out 20 \
--n_iter 4000

# run the model that considers more points rather than a random point (which the thesis did).
python examples/train_resnet.py \
--input_dir data/processed/sample \
--output_dir checkpoints/sample/resnet32/top2000/1e-3 \
--learning_rate 1e-3 \
--n_epoch 2000 \
--per_gpu_batch_size 90000 \
--num_eval_per_epoch 2 \
--weight_decay 1e-6 \
--top_k 2000 \
--hidden_dims_list resnet32 \
--add_shortcut True

# an exampe of tuning HIDEEN_DIMS (see shell codes in ./examples)
TOP_K=2000
LR=1e-3

for HIDDEN_DIMS in resnet32 resnet47
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
```
New methods
------------
1. Learning a distance measure: `notebooks/learn_distance_measure-*.ipynb`.
2. Parametric t-SNE: `notebooks/Parametric t-SNE (Keras).ipynb`.
3. Margin-based methods: the contrastive loss and triplet loss. See `notebooks/Experiments_siamese.ipynb`, `notebooks/analyze_siamese_modeling.ipynb` and `src/models/siamese_triplet`.
4. Calculate level_grade for three types of pairs (mutual neighbors, one-direction neighbors, not neighbors): `notebooks/level_grade_s0.ipynb`.
5. Take top 100 points into consideration: `notebooks/sift_experiment.ipynb`. 
6. Add denoising and maintain absolute distances of all points: `./src/toolkit`. 

Project Organization
------------
```
.
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── artificial3_10_200.csv
│   ├── dev.csv
│   ├── interim
│   │   └── artificial3_10_200.csv
│   ├── processed
│   │   └── sample
│   │       ├── test.csv
│   │       └── train.csv
│   ├── raw
│   ├── readme.md
│   ├── t-sne-2000iters.pth.tar
│   └── train.csv
├── directory.txt
├── docs
│   ├── Makefile
│   ├── commands.rst
│   ├── conf.py
│   ├── getting-started.rst
│   ├── index.rst
│   └── make.bat
├── examples
│   ├── resnet_tuning.sh
│   ├── run_tsne.sh
│   ├── train.py
│   ├── train_resnet.py
│   └── tuning.sh
├── models
│   └── model_zoo.json
├── notebooks
│   ├── Experiments_MNIST.ipynb
│   ├── Experiments_siamese.ipynb
│   ├── Experiments_triplet.ipynb
│   ├── Parametric\ t-SNE\ (Keras).ipynb
│   ├── Parametric\ t-SNE\ (PyTorch).ipynb
│   ├── analysis.ipynb
│   ├── analysis_distance_modeling.ipynb
│   ├── analyze_siamese_modeling.ipynb
│   ├── binary_evaluation.ipynb
│   ├── draw.ipynb
│   ├── inspect_checkpoints.ipynb
│   ├── learn_distance_measure-close-seperate.ipynb
│   ├── learn_distance_measure-multiple-neg.ipynb
│   ├── learn_distance_measure-server.ipynb
│   ├── learn_distance_measure.ipynb
│   ├── learn_distance_measure_sample.ipynb
│   ├── learn_siamese_network.ipynb
│   ├── level_grade_s0.ipynb
│   ├── level_grade_s1.ipynb
│   ├── level_grade_s2.ipynb
│   ├── level_graded_kl_div.ipynb
│   ├── margin_model.pth.tar
│   ├── margin_model_evaluation.ipynb
│   ├── margin_model_evaluation2.ipynb
│   ├── margin_model_evaluation3.ipynb
│   ├── naive_kl.ipynb
│   ├── pre-compute-nn.ipynb
│   ├── sanity_check.ipynb
│   ├── scalability_test-Copy1.ipynb
│   ├── scalability_test.ipynb
│   ├── sift_experiment.ipynb
│   ├── stable.ipynb
│   ├── statistics_of_train_and_test.ipynb
│   ├── trends.ipynb
│   ├── tsne-baseline.ipynb
│   └── visual.png
├── references
│   ├── Thesis_Data_Code
│   │   ├── data.py
│   │   ├── loss.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── settings.py
│   │   ├── similarity.py
│   │   ├── testwithdropout.py
│   │   └── utils.py
│   ├── artificial3_10_200.csv
│   └── tuning-test-code
│       ├── main.py
│       └── sweep.yml
├── reports
│   └── figures
├── requirements.txt
├── samples
│   ├── debug.csv
│   ├── dev.csv
│   └── train.csv
├── setup-cuda
├── setup.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-37.pyc
│   ├── data
│   │   ├── __init__.py
│   │   ├── make_dataset.py
│   │   ├── make_sample.py
│   │   └── utils.py
│   ├── datasets
│   │   └── SIFT.py
│   ├── features
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models
│   │   ├── DenseNetwork
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-37.pyc
│   │   │   │   ├── loss.cpython-37.pyc
│   │   │   │   └── models.cpython-37.pyc
│   │   │   ├── loss.py
│   │   │   └── models.py
│   │   ├── ResNet
│   │   │   ├── ResNet.py
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-37.pyc
│   │   ├── distance_modeling.py
│   │   ├── level_kv_div
│   │   │   ├── binaryTrainer.py
│   │   │   ├── datasets.py
│   │   │   ├── klTrainer.py
│   │   │   ├── network.py
│   │   │   └── utils.py
│   │   ├── predict_model.py
│   │   ├── siamese_triplet
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py
│   │   │   ├── losses.py
│   │   │   ├── metrics.py
│   │   │   ├── networks.py
│   │   │   ├── trainer.py
│   │   │   └── utils.py
│   │   ├── tSNE
│   │   │   ├── __init__.py
│   │   │   └── tSNE.py
│   │   ├── test.py
│   │   ├── utils
│   │   │   ├── Trainer.py
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-37.pyc
│   │   │   │   └── distance.cpython-37.pyc
│   │   │   ├── data.py
│   │   │   ├── distance.py
│   │   │   └── loss.py
│   │   └── vanilla
│   │       ├── __init__.py
│   │       ├── data.py
│   │       ├── loss.py
│   │       ├── main.py
│   │       ├── model.py
│   │       ├── settings.py
│   │       ├── similarity.py
│   │       └── utils.py
│   ├── toolkit
│   │   ├── MNIST
│   │   │   ├── processed
│   │   │   │   ├── test.pt
│   │   │   │   └── training.pt
│   │   │   └── raw
│   │   │       ├── t10k-images-idx3-ubyte
│   │   │       ├── t10k-images-idx3-ubyte.gz
│   │   │       ├── t10k-labels-idx1-ubyte
│   │   │       ├── t10k-labels-idx1-ubyte.gz
│   │   │       ├── train-images-idx3-ubyte
│   │   │       ├── train-images-idx3-ubyte.gz
│   │   │       ├── train-labels-idx1-ubyte
│   │   │       └── train-labels-idx1-ubyte.gz
│   │   ├── __init__.py
│   │   ├── final.py
│   │   ├── learn.py
│   │   ├── network.py
│   │   ├── score.py
│   │   └── tune.py
│   ├── utils.py
│   └── visualization
│       ├── __init__.py
│       └── visualize.py
├── src.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── test_environment.py
└── tox.ini

37 directories, 153 files

```
--------
