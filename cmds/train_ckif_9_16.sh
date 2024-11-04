SAVE_DIR=exp/train_ckif_9_16; mkdir -p $SAVE_DIR; cp tools/ckif_train.py $SAVE_DIR/; python tools/ckif_train.py --config configs/FederatedLearning/PtoR_config.yml --data_paths TPL/client_9,TPL/client_10,TPL/client_11,TPL/client_12,TPL/client_13,TPL/client_14,TPL/client_15,TPL/client_16 --num_round 50 --num_local_epochs 5 --num_clients 8 --num_finetune 10 --num_gpus 4 --softmax_temp 1.5 --save_dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/train.log

# use CUDA_VISIBLE_DEVICES=X,X,X,X if needed.
