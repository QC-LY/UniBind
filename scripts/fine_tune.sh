cd ../
CUDA_VISIBLE_DEVICES=0 python train.py --train_dataset_dir ./datasets/xx/train_dataset \
 --eval_dataset_dir ./datasets/xx/eval_dataset \
 --test_dataset_dir ./datasets/xx/test_dataset \
 --train_data_path ./datasets/xx/train_data.json \
 --eval_data_path ./datasets/xx/eval_data.json \
 --test_data_path ./datasets/xx/test_data.json \
 --centre_embeddings_path ./centre_embs/xx_center_embeddings.pkl \
 --pretrain_weights ./ckpts/pretrained_weights.pt \
 --output_dir ./outputs/image_xx \
 --modality image \
 --train_batch_size 1024 \
 --val_batch_size 256 \
 --num_workers 8 \
 --learning_rate 5e-4 \
 --num_train_epochs 15 \
 --eval_steps 1000 \
 --seed 1234 \