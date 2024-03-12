cd ../
CUDA_VISIBLE_DEVICES=0 python infer.py --test_dataset_dir ./datasets/xx/test_dataset \
 --test_data_path ./datasets/xx/test_data.json \
 --centre_embeddings_path ./centre_embs/xx_center_embeddings.pkl \
 --pretrain_weights ./ckpts/pretrained_weights.pt \
 --output_dir ./outputs/image_xx \
 --modality image \
 --val_batch_size 16 \
 --num_workers 0 \
 --seed 1234 \