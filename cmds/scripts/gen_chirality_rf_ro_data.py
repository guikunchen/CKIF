from rdkit import Chem
import os
import argparse
from tqdm import tqdm
import multiprocessing
import pandas as pd
from rdkit import RDLogger
import re
import random

# python configs/scripts/gen_chemical_data.py -targets ./dataset/USPTO_50K_PtoR_aug1/train/tgt-train.txt -sources ./dataset/USPTO_50K_PtoR_aug1/train/src-train.txt
# python configs/scripts/gen_chemical_data.py -input ./dataset/USPTO_50K_PtoR_aug1 -output ./dataset/USPTO_50K_PtoR_aug1/chirality
# python cmds/scripts/data/gen_chirality_rf_ro_data.py -input ./dataset/USPTO-MIT_PtoR_aug1 -output ./dataset/USPTO-MIT_PtoR_aug1/tmp -need_type ringopening
# python cmds/scripts/gen_chirality_rf_ro_data.py -input ./dataset/USPTO-MIT_PtoR_aug1 -output ./dataset/mixed/client_7

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def canonicalize_smiles_clear_map(smiles,return_max_frag=True):
    mol = Chem.MolFromSmiles(smiles,sanitize=True)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            if return_max_frag:
                return '',''
            else:
                return ''
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [Chem.MolFromSmiles(smiles,sanitize=True) for smiles in sub_smi]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms())) for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(sorted(sub_mol_size,key=lambda x:x[1],reverse=True)[0][0],return_max_frag=False)
            else:
                return smi, ''
        else:
            return smi
    else:
        if return_max_frag:
            return '',''
        else:
            return ''


def main(opt):
    for dataset in ['train', 'val', 'test']:
        with open(os.path.join(opt.input, dataset, f'tgt-{dataset}.txt'), 'r') as f:
            tgt_lines = f.readlines()
            targets = [''.join(line.strip().split(' ')) for line in tgt_lines]
            pool = multiprocessing.Pool(processes=opt.process_number)
            targets = pool.map(func=canonicalize_smiles_clear_map, iterable=targets)
            pool.close()
            pool.join()
        ground_truth = targets
        with open(os.path.join(opt.input, dataset, f'src-{dataset}.txt'), "r") as f:
            src_lines = f.readlines()
            ras_src_smiles = [''.join(line.strip().split(' ')) for line in src_lines]

        type2indices = {'chirality': [], 'ringopening': [], 'ringformation': []}
        
        for i in tqdm(range(len(ground_truth))):
            pro_mol = Chem.MolFromSmiles(ras_src_smiles[i])
            rea_mol = Chem.MolFromSmiles(ground_truth[i][0])  # ground_truth[i][1] is MaxFrag
            pro_ringinfo = pro_mol.GetRingInfo()
            rea_ringinfo = rea_mol.GetRingInfo()
            pro_ringnum = pro_ringinfo.NumRings()
            rea_ringnum = rea_ringinfo.NumRings()
            # size = len(rea_mol.GetAtoms()) - len(pro_mol.GetAtoms())
            # if (int(ras_src_smiles[i].count("@") > 0) + int(ground_truth[i][0].count("@") > 0)) == 1:
            if ras_src_smiles[i].count("@") > 0 or ground_truth[i][0].count("@") > 0:
                type2indices["chirality"].append(i)
            if pro_ringnum < rea_ringnum:
                type2indices["ringopening"].append(i)
            if pro_ringnum > rea_ringnum:
                type2indices["ringformation"].append(i)

        print(len(tgt_lines), len(ground_truth), len(type2indices["chirality"]), len(type2indices["ringopening"]), len(type2indices["ringformation"]))
        os.makedirs(os.path.join(opt.output, dataset), exist_ok=True)
        
        if opt.need_type == 'chirality':
            with open(os.path.join(opt.output, dataset, f'tgt-{dataset}.txt'), 'w') as f:
                f.writelines([tgt_lines[i] for i in type2indices["chirality"]])
            with open(os.path.join(opt.output, dataset, f'src-{dataset}.txt'), 'w') as f:
                f.writelines([src_lines[i] for i in type2indices["chirality"]])
        elif opt.need_type == 'ringformation':
            with open(os.path.join(opt.output, dataset, f'tgt-{dataset}.txt'), 'w') as f:
                f.writelines([tgt_lines[i] for i in type2indices["ringformation"]])
            with open(os.path.join(opt.output, dataset, f'src-{dataset}.txt'), 'w') as f:
                f.writelines([src_lines[i] for i in type2indices["ringformation"]])
        elif opt.need_type == 'ringopening':
            # We randomly sampled a subset due to too much samples. So we use the same copy here.
            with open(os.path.join(opt.output, dataset, f'tgt-{dataset}.txt'), 'w') as f:
                f.writelines(open(os.path.join('./dataset/mixed/client_8', dataset, f'tgt-{dataset}.txt')).readlines())
            with open(os.path.join(opt.output, dataset, f'src-{dataset}.txt'), 'w') as f:
                f.writelines(open(os.path.join('./dataset/mixed/client_8', dataset, f'src-{dataset}.txt')).readlines())
            # The original code is here if you want to do a resampling.
            # sampled_indices = random.sample(type2indices["ringopening"], len(type2indices["ringopening"]) // 5)
            # with open(os.path.join(opt.output, dataset, f'tgt-{dataset}.txt'), 'w') as f:
            #     f.writelines([tgt_lines[i] for i in sampled_indices])
            # with open(os.path.join(opt.output, dataset, f'src-{dataset}.txt'), 'w') as f:
            #     f.writelines([src_lines[i] for i in sampled_indices])
        else:
            assert False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='select samples',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', type=str, required=True, help="Path to file")
    parser.add_argument('-output', type=str, required=True, help="Path to file")
    parser.add_argument('-need_type', type=str, required=True)
    parser.add_argument('-process_number', type=int, default=multiprocessing.cpu_count())
    opt = parser.parse_args()
    main(opt)
