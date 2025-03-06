import subprocess
import argparse
import yaml
import os
import torch
import time
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def update_yml_data(fpath1, fpath2, data_path):
    with open(fpath1, 'r') as fp:
        old_cfg = yaml.full_load(fp)
    old_cfg['data']['corpus_1']['path_src'] = './dataset/{}/train/src-train.txt'.format(data_path)
    old_cfg['data']['corpus_1']['path_tgt'] = './dataset/{}/train/tgt-train.txt'.format(data_path)
    with open(fpath2, 'w') as fp:
        yaml.dump(old_cfg, fp, indent=2)


def update_yml_cfg(fpath1, fpath2, cfg):
    with open(fpath1, 'r') as fp:
        old_cfg = yaml.full_load(fp)
    old_cfg.update(cfg)
    with open(fpath2, 'w') as fp:
        yaml.dump(old_cfg, fp, indent=2)


def compute_weights_val(args, in_client_idx, client_steps, n_beam=10, ckpt_weights=None):
    def _softmax(vec, temp=1., mask_idx=None):
        vec = np.array(vec) / temp
        if mask_idx is not None:
            vec[mask_idx] = -9999999.
        exponential = np.exp(vec)
        probabilities = exponential / np.sum(exponential)
        return probabilities.tolist()

    cmds = []
    for client_idx in range(args.num_clients):
        if client_idx == in_client_idx:
            continue
        cmds.append('CUDA_VISIBLE_DEVICES={} onmt_translate -config configs/translate/translate.yml -model {} -src ./dataset/{}/val/src-val.txt -output {}'.format(client_idx % args.num_gpus, os.path.join(args.save_dir, f'client_{client_idx}/p2r_step_{client_steps[client_idx]}.pt'), args.data_paths[in_client_idx], os.path.join(args.save_dir, f'client_{client_idx}/results_val_{in_client_idx}.txt')))
    exit_code = subprocess.call(' & '.join(cmds), shell='bash')
    while True:
        flag_done = True
        for client_idx in range(args.num_clients):
            if client_idx == in_client_idx:
                continue
            if time.time() - os.path.getmtime(os.path.join(args.save_dir, f'client_{client_idx}/results_val_{in_client_idx}.txt')) < 10:
                flag_done = False
        if flag_done:
            time.sleep(10); break
        else:
            time.sleep(10)
    
    scores = []
    for client_idx in range(args.num_clients):
        if client_idx == in_client_idx:
            scores.append(-999.)
            continue
        with open('./dataset/{}/val/tgt-val.txt'.format(args.data_paths[in_client_idx]), 'r') as fp:
            gt_smiles = [line.strip().replace(' ', '') for line in fp.readlines()]
        with open(os.path.join(args.save_dir, f'client_{client_idx}/results_val_{in_client_idx}.txt'), 'r') as fp:
            pred_smiles = [line.strip().replace(' ', '') for line in fp.readlines()]
        # assert len(gt_smiles) * n_beam == len(pred_smiles)
        while len(gt_smiles) * n_beam != len(pred_smiles):  # wait for translate
            time.sleep(10)
            with open(os.path.join(args.save_dir, f'client_{client_idx}/results_val_{in_client_idx}.txt'), 'r') as fp:
                pred_smiles = [line.strip().replace(' ', '') for line in fp.readlines()]
        cur_scores = []
        for data_idx in range(len(gt_smiles)):
            gt_mol = Chem.MolFromSmiles(gt_smiles[data_idx])
            preds = pred_smiles[n_beam * data_idx: n_beam * (data_idx + 1)]
            score = []
            for pred in preds:
                pred_mol = Chem.MolFromSmiles(pred)
                if pred_mol is not None:
                    maccs_gt = AllChem.GetMACCSKeysFingerprint(gt_mol)
                    maccs_pred = AllChem.GetMACCSKeysFingerprint(pred_mol)
                    score.append(DataStructs.TanimotoSimilarity(maccs_pred, maccs_gt))
                else:  # illegal mol
                    score.append(0.)
                
            cur_scores.extend(score)
        scores.append(sum(cur_scores) / len(cur_scores))
    
    print('avg scores:', scores, end='\t')
    if args.agg_self_w == 'n_sample':
        in_client_w = ckpt_weights[in_client_idx]
    elif args.agg_self_w == 'n_client':
        in_client_w = 1. / args.num_clients
    else:
        in_client_w = float(args.agg_self_w)
    weights = _softmax([i for i in scores], temp=args.softmax_temp, mask_idx=in_client_idx)
    weights = [i * (1 - in_client_w) for i in weights]
    weights[in_client_idx] = in_client_w

    return weights


def aggregate_model(args, round_idx, ckpt_weights, client_steps):
    model_files = []
    for client_idx in range(args.num_clients):
        model_files.append(os.path.join(args.save_dir, f'client_{client_idx}/p2r_step_{client_steps[client_idx]}.pt'))
    while True:
        flag_exist = True
        for f in model_files:
            if not os.path.exists(f): flag_exist = False
        if flag_exist:
            time.sleep(10); break
        else:
            time.sleep(10)
    for client_idx in range(args.num_clients):
        weights = compute_weights_val(args, client_idx, client_steps, ckpt_weights=ckpt_weights)
        print(round_idx, weights, flush=True)
        weighted_m = None
        for i, model_file in enumerate(model_files):
            m = torch.load(model_file, map_location='cpu')
            if i == 0:
                weighted_m = m
                for k in weighted_m['model']:
                    weighted_m['model'][k].mul_(weights[i])
                for k in weighted_m['generator']:
                    weighted_m['generator'][k].mul_(weights[i])
            else:
                for (k, v) in m['model'].items():
                    weighted_m['model'][k].add_(v * weights[i])
                for (k, v) in m['generator'].items():
                    weighted_m['generator'][k].add_(v * weights[i])
        torch.save(weighted_m, model_files[client_idx][:-3] + '_aligned.pt')
    
    for i, model_file in enumerate(model_files):
        if round_idx != args.num_round - 1:
            os.remove(model_file)
            os.system('rm {}'.format(os.path.join(args.save_dir, f'client_{i}/results_val_*.txt')))


def main(args):
    client_data_lens = []
    for data_path in args.data_paths:
        with open('./dataset/{}/train/src-train.txt'.format(data_path), 'r') as fp:
            client_data_lens.append(len(fp.readlines()))
    client_ckpt_w = [i / sum(client_data_lens) for i in client_data_lens]

    for round_idx in range(args.cur_round, args.num_round):
        if round_idx == 0:  # the first round
            cmds = []
            for client_idx in range(args.num_clients):
                update_yml_data(args.config, os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml'), args.data_paths[client_idx])
                train_steps = int(client_data_lens[client_idx] * args.num_local_epochs / args.num_local_bs)
                update_yml_cfg(os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml'), os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml'), {'save_model': os.path.join(args.save_dir, f'client_{client_idx}/p2r'), 'save_checkpoint_steps': train_steps, 'train_steps': train_steps, 'report_every': train_steps // 10})
                cmds.append('CUDA_VISIBLE_DEVICES={} onmt_train -config {}'.format(client_idx % args.num_gpus, os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml')))
            exit_code = subprocess.call(' & '.join(cmds), shell='bash')
            aggregate_model(args, round_idx, client_ckpt_w, [int(data_len * args.num_local_epochs / args.num_local_bs) for data_len in client_data_lens])
        elif round_idx < args.num_round - args.num_finetune:
            cmds = []
            for client_idx in range(args.num_clients):
                train_steps = int(client_data_lens[client_idx] * args.num_local_epochs / args.num_local_bs)
                update_yml_cfg(os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml'), os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml'), {'train_from': os.path.join(args.save_dir, 'client_{}/p2r_step_{}_aligned.pt'.format(client_idx, train_steps)), 'save_model': os.path.join(args.save_dir, f'client_{client_idx}/p2r')})
                cmds.append('CUDA_VISIBLE_DEVICES={} onmt_train -config {}'.format(client_idx % args.num_gpus, os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml')))
            exit_code = subprocess.call(' & '.join(cmds), shell='bash')
            aggregate_model(args, round_idx, client_ckpt_w, [int(data_len * args.num_local_epochs / args.num_local_bs) for data_len in client_data_lens])
        elif round_idx < args.num_round and round_idx == args.num_round - args.num_finetune:
            cmds = []
            for client_idx in range(args.num_clients):
                train_steps = int(client_data_lens[client_idx] * args.num_local_epochs / args.num_local_bs)
                update_yml_cfg(os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml'), os.path.join(args.save_dir, f'client_{client_idx}/cfg_fine_tuned.yml'), {'train_from': os.path.join(args.save_dir, 'client_{}/p2r_step_{}_aligned.pt'.format(client_idx, train_steps)), 'save_checkpoint_steps': int(train_steps * args.num_finetune), 'train_steps': int(train_steps * args.num_finetune), 'save_model': os.path.join(args.save_dir, f'client_{client_idx}/p2r'), 'report_every': 1000})
                # update_yml_cfg(os.path.join(args.save_dir, f'client_{client_idx}/cfg.yml'), os.path.join(args.save_dir, f'client_{client_idx}/cfg_fine_tuned.yml'), {'train_from': os.path.join(args.save_dir, 'client_{}/p2r_step_{}_aligned.pt'.format(client_idx, train_steps)), 'save_checkpoint_steps': train_steps, 'train_steps': int(train_steps * args.num_finetune), 'keep_checkpoint': min(10, args.num_finetune), 'save_model': os.path.join(args.save_dir, f'client_{client_idx}/p2r'), 'report_every': 1000})
                cmds.append('CUDA_VISIBLE_DEVICES={} onmt_train -config {}'.format(client_idx % args.num_gpus, os.path.join(args.save_dir, f'client_{client_idx}/cfg_fine_tuned.yml')))
            exit_code = subprocess.call(' & '.join(cmds), shell='bash')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CKIF Hyperparameters")
    parser.add_argument('--cur_round', type=int, default=0,
                        help='Current round number to start/resume training (default: 0)')
    parser.add_argument('--num_round', type=int, default=50,
                        help='Total number of federated communication rounds (default: 50)')
    parser.add_argument('--num_local_epochs', type=int, default=5,
                        help='Number of local training epochs per client (default: 5)')
    parser.add_argument('--num_local_bs', type=int, default=64,
                        help='Local batch size for client training (default: 64)')
    parser.add_argument('--num_clients', type=int, default=4,
                        help='Total number of participating clients (default: 4)')
    parser.add_argument('--num_finetune', type=int, default=10,
                        help='Number of fine-tuning epochs (default: 10)')
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='Number of GPUs to use for training (default: 4)')
    parser.add_argument('--softmax_temp', type=float, default=1.5,
                        help='Temperature parameter for softmax smoothing (default: 1.5)')
    parser.add_argument('--agg_self_w', type=str, default='n_client',
                        help='Weighting scheme for model aggregation [options: "n_client", "equal"] (default: "n_client")')
    parser.add_argument('--save_dir', type=str, default='./exp/tmp/',
                        help='Directory to save logs, models, and results (default: "./exp/tmp/")')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file defining model and training parameters')
    parser.add_argument('--data_paths', type=str, required=True,
                        help='Comma-separated paths to client-specific datasets (required)')

    args = parser.parse_args()
    args.data_paths = args.data_paths.strip().split(',')
    assert len(args.data_paths) == args.num_clients
    os.makedirs(os.path.join(args.save_dir, 'server'), exist_ok=True)
    for client_idx in range(args.num_clients):
        os.makedirs(os.path.join(args.save_dir, f'client_{client_idx}'), exist_ok=True)
    print(args)
    main(args)
