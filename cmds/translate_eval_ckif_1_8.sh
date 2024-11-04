infer_ckif_1_8_50e () {
    SAVE_DIR=$1/client_0
    MODEL_NAME=p2r_step_7430.pt
    CLI=1
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/mixed/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/mixed/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/mixed/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_1
    MODEL_NAME=p2r_step_3530.pt
    CLI=2
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/mixed/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/mixed/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/mixed/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_2
    MODEL_NAME=p2r_step_6130.pt
    CLI=3
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/mixed/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/mixed/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/mixed/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_3
    MODEL_NAME=p2r_step_4670.pt
    CLI=4
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/mixed/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/mixed/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/mixed/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_4
    MODEL_NAME=p2r_step_9460.pt
    CLI=5
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/mixed/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/mixed/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/mixed/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_5
    MODEL_NAME=p2r_step_5580.pt
    CLI=6
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/mixed/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/mixed/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/mixed/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_6
    MODEL_NAME=p2r_step_8120.pt
    CLI=7
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/mixed/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/mixed/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/mixed/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_7
    MODEL_NAME=p2r_step_6630.pt
    CLI=8
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/mixed/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/mixed/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/mixed/client_$CLI/test/src-test.txt
}

infer_ckif_1_8_50e ./exp/train_ckif_1_8
