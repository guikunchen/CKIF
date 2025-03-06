# Data preprocessing
## Raw files
All chemical reaction data used in this work are publicly available. The USPTO-50K dataset can be found at https://www.dropbox.com/sh/6ideflxcakrak10/AADTbFBC0F8ax55-z-EDgrIza. The USPTO-MIT dataset can be found at https://github.com/wengong-jin/nips17-rexgen/blob/master/USPTO/data.zip. The USPTO 1k TPL dataset can be found at https://ibm.ent.box.com/v/MappingChemicalReactions.

Download, unzip, and rename them. The dataset should be like this:
```
│dataset/
├──USPTO_TPL/
│  ├── uspto_1k_TPL_test.tsv
│  ├── uspto_1k_TPL_train_valid.tsv
├──USPTO_50K/
│  ├── raw_test.csv
│  ├── raw_train.csv
│  ├── raw_val.csv
├──USPTO-MIT/
│  ├── test.txt
│  ├── train.txt
│  ├── val.txt
```

## mixed
```shell
python cmds/scripts/generate_PtoR_data.py -dataset USPTO-MIT -canonical

python cmds/scripts/generate_PtoR_data.py -dataset USPTO_50K -canonical

python cmds/scripts/generate_PtoR_data_clients_50k_class.py

python cmds/scripts/gen_chirality_rf_ro_data.py -input ./dataset/USPTO_50K_PtoR_aug1 -output ./dataset/mixed_reproduced/client_6 -need_type chirality

python cmds/scripts/gen_chirality_rf_ro_data.py -input ./dataset/USPTO-MIT_PtoR_aug1 -output ./dataset/mixed_reproduced/client_7 -need_type ringformation

python cmds/scripts/gen_chirality_rf_ro_data.py -input ./dataset/USPTO-MIT_PtoR_aug1 -output ./dataset/mixed_reproduced/client_8 -need_type ringopening
```

## TPL
```shell
python cmds/scripts/tpl_tsv2csv.py

python cmds/scripts/gen_tpl_client_splits.py

python cmds/scripts/generate_PtoR_data_clients_TPL_class.py 
```

# Contaminated data
```shell
python cmds/scripts/gen_contaminated_data.py
```
