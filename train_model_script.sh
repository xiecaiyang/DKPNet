# train model_dkpnet
python main_train.py --train_dir /home/xcy/dataset/dataset_denoise/BSD300/ --test_dir /home/xcy/dataset/dataset_denoise/BSD200/ --ck ./checkpoint/dkpnet_nl30/ --noise_level 30 --kp_size 3 --n1_resblocks 9 --n2_resblocks 2 --n_feats 64 --cuda 
