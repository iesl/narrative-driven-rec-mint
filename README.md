This code is intended to aid anonymous reviews, it isnt setup for allowin reproducibility efforts yet.

Some files which may be of interest:

`src/pre_process/yelppoi_prompts.py`: Prompts used for generating synthetic queries and making Grounded LLM predictions.

`src/pre_process/pre_proc_yelppoi.py`: Pre-processing the user-item interactions and generating synthetic narrative queries.

`src/learning/main_fsim.py`: Starting point for examining bi-encoder implementations and training.

`src/learning/main_ce.py`: Starting point for examining cross-encoder implementations and training.

`config/models_config/yelppoi`: Config files with hyperparameters for bi-encoders and cross-encoders. `recinpars` refers to the bi-encoder and `recinparsce` refers to the cross-encoder.
