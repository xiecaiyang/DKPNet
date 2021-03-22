#evaluate model_dkpnet, make sure the network setting is the same as in the pretrained model
python main_eval.py --test_dir /home/xcy/dataset/dataset_denoise/est_BSD200_crop/ --result_dir ./eval_result/dkpnet_BSD200_nl30 --resume ./checkpoint/pretrained_models/model_dkpnet_nl30.pth --kp_size 3 --n1_resblocks 9 --n2_resblocks 2 --noise_level 30 --cuda
