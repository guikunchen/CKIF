# Chemical knowledge-informed framework for privacy-aware retrosynthesis learning

This is the official implementation of "[Chemical knowledge-informed framework for privacy-aware retrosynthesis learning](https://arxiv.org/abs/2502.19119)".

## Environment Preparation

Please make sure you have installed anaconda/miniconda.

```shell
git clone https://github.com/guikunchen/CKIF.git
conda create -n ckif python=3.7
conda activate ckif
conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install pandas==1.3.5 textdistance==4.2.2 portalocker==2.7.0
conda install rdkit=2020.09.1.0 -c rdkit
cd third_party
pip install -e .  # install OpenNMT-py
```


## Data

If you clone this repo, the datasets should be like this:

```
│dataset/
├──mixed/
│  ├── client_1
│  ├── client_2
│  ├── ......
├──TPL/
│  ├── client_9
│  ├── client_10
│  ├── ......
├──vocab/
│  ├── example.vocab.src
│  ├── empty.txt
```

For the "mixed" folder, clients 1-6 are from USPTO-50K; clients 7-8 are from USPTO-MIT. For the "TPL" folder, all clients are from USPTO 1k TPL. See [DATASET.md](DATASET.md) for the detailed dataset acquisition.

## Training and Evaluation

Here is an example of how to train and evaluate the model, given 4 clients from USPTO-50K.

```shell
bash cmds/train_ckif_1_4.sh
bash cmds/translate_eval_ckif_1_4.sh
# after that, you can find the evaluation results in ./exp/client_X/p2r_step_X.pt.c_X_eval.results.txt
```

See folder [cmds](cmds/) for more scripts for training and evaluation.

## Parameter Configuration Instruction

```shell
python tools/ckif_train.py --help
```

## Core Functions

For reproducibility, using a single machine is sufficient for monitoring the federated learning process. CKIF can be simply applied in real-world applications by exchanging only the model parameters (without sharing data across clients).

### Monitoring local learning
```python
cmds = []
for client_idx in range(args.num_clients):
    train_steps = int(client_data_lens[client_idx] * args.num_local_epochs / args.num_local_bs)
    update_yml_cfg(os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml'), os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml'), {'train_from': os.path.join(args.save_dir, 'client_{}/p2r_step_{}_aligned.pt'.format(client_idx, train_steps)), 'save_model': os.path.join(args.save_dir, f'client_{client_idx}/p2r')})
    cmds.append('CUDA_VISIBLE_DEVICES={} onmt_train -config {}'.format(client_idx % args.num_gpus, os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml')))
exit_code = subprocess.call(' & '.join(cmds), shell='bash')
```

This preserves privacy through client-local data processing and model training, ensuring raw user data never leaves client devices. It can be further enhanced via encrypted parameter exchange (not implemented here for saving computing resources).

### Model aggregation
See the aggregate_model function in [ckif_train.py](tools/ckif_train.py).

### Chemical knowledge-informed weighting
See the compute_weights_val function in [ckif_train.py](tools/ckif_train.py).


## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[Root-aligned SMILES](https://github.com/otori-bird/retrosynthesis), [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

## Contact

Any comments, please email: guikunchen@gmail.com.
