export CUDA_VISIBLE_DEVICES=3

python main_evaluation.py \
--cfd_model=Transolver \
--data_dir /content/drive/My Drive/transolver/mlcfd_data/training_data \
--save_dir /content/drive/My Drive/transolver/mlcfd_data/preprocessed_data \
