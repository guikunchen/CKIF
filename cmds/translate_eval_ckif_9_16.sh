infer_ckif_9_16_50e () {
    SAVE_DIR=$1/client_0
    MODEL_NAME=p2r_step_10390.pt
    CLI=9
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/TPL/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/TPL/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/TPL/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_1
    MODEL_NAME=p2r_step_10370.pt
    CLI=10
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/TPL/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/TPL/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/TPL/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_2
    MODEL_NAME=p2r_step_10430.pt
    CLI=11
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/TPL/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/TPL/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/TPL/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_3
    MODEL_NAME=p2r_step_10260.pt
    CLI=12
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/TPL/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/TPL/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/TPL/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_4
    MODEL_NAME=p2r_step_10360.pt
    CLI=13
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/TPL/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/TPL/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/TPL/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_5
    MODEL_NAME=p2r_step_10500.pt
    CLI=14
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/TPL/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/TPL/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/TPL/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_6
    MODEL_NAME=p2r_step_10400.pt
    CLI=15
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/TPL/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/TPL/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/TPL/client_$CLI/test/src-test.txt
    SAVE_DIR=$1/client_7
    MODEL_NAME=p2r_step_10530.pt
    CLI=16
    CUDA_VISIBLE_DEVICES=2 onmt_translate -config configs/translate/translate.yml -batch_size 8192 -src ./dataset/TPL/client_$CLI/test/src-test.txt -model $SAVE_DIR/$MODEL_NAME -output $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt
    python tools/score.py -detailed -raw -beam_size 10 -n_best 10 \
        -targets ./dataset/TPL/client_$CLI/test/tgt-test.txt \
        -predictions $SAVE_DIR/$MODEL_NAME.c_$CLI.results.txt \
        -source ./dataset/TPL/client_$CLI/test/src-test.txt
}

infer_ckif_9_16_50e ./exp/train_ckif_9_16
