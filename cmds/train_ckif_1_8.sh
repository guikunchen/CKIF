SAVE_DIR=exp/train_ckif_1_8; mkdir -p $SAVE_DIR; cp tools/ckif_train.py $SAVE_DIR/; python tools/ckif_train.py --config configs/FederatedLearning/PtoR_config.yml --data_paths mixed/client_1,mixed/client_2,mixed/client_3,mixed/client_4,mixed/client_5,mixed/client_6,mixed/client_7,mixed/client_8 --num_round 50 --num_local_epochs 5 --num_clients 8 --num_finetune 10 --num_gpus 4 --softmax_temp 1.5 --save_dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/train.log

# use CUDA_VISIBLE_DEVICES=X,X,X,X if needed.
