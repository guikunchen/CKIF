import numpy as np
import pandas as pd
import argparse
import os
import re
import random
import textdistance
import multiprocessing
from rdkit import Chem
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


def get_cano_map_number(smi,root=-1):
    atommap_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi,root=root))
    cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
    correct_mapped = [canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol() for i,index in enumerate(cano2atommapIdx)]
    atom_number = len(canonical_mol.GetAtoms())
    if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
        cano2atommapIdx = [0] * atom_number
        atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)
        if len(atommap2canoIdx) != atom_number:
            return None
        for i, index in enumerate(atommap2canoIdx):
            cano2atommapIdx[index] = i
    id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]

    return [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]


def get_root_id(mol,root_map_number):
    root = -1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            root = i
            break
    return root
    # root = -1
    # for i, atom in enumerate(mol.GetAtoms()):
    #     if atom.GetAtomMapNum() == root_map_number:
    #         return i


"""multiprocess"""
def preprocess(save_dir, reactants, products,set_name, augmentation=1, reaction_types=None,root_aligned=True,character=False, processes=-1):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = [ {
        "reactant":i,
        "product":j,
        "augmentation":augmentation,
        "root_aligned":root_aligned,
    }  for i,j in zip(reactants,products)]
    src_data = []
    tgt_data = []
    skip_dict = {
        'invalid_p':0,
        'invalid_r':0,
        'small_p':0,
        'small_r':0,
        'error_mapping':0,
        'error_mapping_p':0,
        'empty_p':0,
        'empty_r':0,
    }
    processes = multiprocessing.cpu_count() if processes < 0 else processes
    pool = multiprocessing.Pool(processes=processes)
    results = pool.map(func=multi_process,iterable=data)
    pool.close()
    pool.join()
    edit_distances = []
    for result in tqdm(results):
        if result['status'] != 0:
            skip_dict[result['status']] += 1
            continue
        if character:
            for i in range(len(result['src_data'])):
                result['src_data'][i] = " ".join([char for char in "".join(result['src_data'][i].split())])
            for i in range(len(result['tgt_data'])):
                result['tgt_data'][i] = " ".join([char for char in "".join(result['tgt_data'][i].split())])
        edit_distances.append(result['edit_distance'])
        src_data.extend(result['src_data'])
        tgt_data.extend(result['tgt_data'])
    print("Avg. edit distance:", np.mean(edit_distances))
    print('size', len(src_data))
    for key,value in skip_dict.items():
        print(f"{key}:{value},{value/len(reactants)}")
    if augmentation != 999:
        with open(
                os.path.join(save_dir, 'src-{}.txt'.format(set_name)), 'w') as f:
            for src in src_data:
                f.write('{}\n'.format(src))

        with open(
                os.path.join(save_dir, 'tgt-{}.txt'.format(set_name)), 'w') as f:
            for tgt in tgt_data:
                f.write('{}\n'.format(tgt))
    return src_data,tgt_data


def multi_process(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status":0,
        "src_data":[],
        "tgt_data":[],
        "edit_distance":0,
    }
    # if ",".join(rids) != ",".join(pids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    # if len(set(rids)) != len(rids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    # if len(set(pids)) != len(pids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if len(pro_mol.GetAtoms()) == 1:
        return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if data['root_aligned']:
            reversable = False  # no shuffle
            # augmentation = 100
            if augmentation == 999:
                product_roots = pro_atom_map_numbers
                times = len(product_roots)
            else:
                product_roots = [-1]
                # reversable = len(reactant) > 1

                max_times = len(pro_atom_map_numbers)
                times = min(augmentation, max_times)
                if times < augmentation:  # times = max_times
                    product_roots.extend(pro_atom_map_numbers)
                    product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
                else:  # times = augmentation
                    while len(product_roots) < times:
                        product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                        # pro_atom_map_numbers.remove(product_roots[-1])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == augmentation
                if reversable:
                    times = int(times / 2)
            # candidates = []
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
                used_indices = []
                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in rea_map_number:
                            rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                            rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
                aligned_reactants = [item[0] for item in sorted_reactants]
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)

                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)

                if reversable:
                    aligned_reactants.reverse()
                    reactant_smi = ".".join(aligned_reactants)
                    product_tokens = smi_tokenizer(pro_smi)
                    reactant_tokens = smi_tokenizer(reactant_smi)
                    return_status['src_data'].append(product_tokens)
                    return_status['tgt_data'].append(reactant_tokens)
            assert len(return_status['src_data']) == data['augmentation']
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(pro_atom_map_numbers)) > 0 ])
            return_status['src_data'].append(smi_tokenizer(cano_product))
            return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation-1)):
                pro_smi = Chem.MolToSmiles(pro_mol,doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol,doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
                return_status['tgt_data'].append(smi_tokenizer(rea_smi))
        edit_distances = []
        for src,tgt in zip(return_status['src_data'],return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(),tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        type=str,
                        default='USPTO_TPL')
    parser.add_argument("-augmentation",type=int,default=1)
    parser.add_argument("-num_clients",type=int,default=8)
    parser.add_argument("-seed",type=int,default=33)
    parser.add_argument("-processes",type=int,default=-1)
    parser.add_argument("-test_only", action="store_true")
    parser.add_argument("-train_only", action="store_true")
    parser.add_argument("-test_except", action="store_true")
    parser.add_argument("-validastrain", action="store_true")
    parser.add_argument("-character", action="store_true")
    parser.add_argument("-canonical", action="store_true")
    args = parser.parse_args()
    print('preprocessing dataset {}...'.format(args.dataset))
    assert args.dataset in ['USPTO_TPL']
    datasets = ['train', 'val', 'test']
    args.canonical = True
    print(args)
    
    random.seed(args.seed)
    datadir = './dataset/{}'.format(args.dataset)
    for i, data_set in enumerate(datasets):
        csv_path = f"{datadir}/raw_{data_set}.csv"
        csv = pd.read_csv(csv_path)
        reaction_list = list(csv["reactants>reagents>production"])

        # random.shuffle(reaction_list)
        reaction_type_list = list(csv["class"])
        reactant_smarts_list = list(
            map(lambda x: x.split('>')[0], reaction_list))
        reactant_smarts_list = list(
            map(lambda x: x.split(' ')[0], reactant_smarts_list))
        reagent_smarts_list = list(
            map(lambda x: x.split('>')[1], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split('>')[2], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split(' ')[0], product_smarts_list))  # remove ' |f:1...'
        print("Total Data Size", len(reaction_list))

        # reaction_class_list = list(map(lambda x: int(x) - 1, csv['class']))
        sub_react_list = reactant_smarts_list
        sub_prod_list = product_smarts_list
        sub_type_list = reaction_type_list
        
        # duplicate multiple product reactions into multiple ones with one product each
        multiple_product_indices = [i for i in range(len(sub_prod_list)) if "." in sub_prod_list[i]]
        for index in multiple_product_indices:
            products = sub_prod_list[index].split(".")
            for product in products:
                sub_react_list.append(sub_react_list[index])
                sub_type_list.append(sub_type_list[index])
                sub_prod_list.append(product)
        for index in multiple_product_indices[::-1]:
            del sub_react_list[index]
            del sub_prod_list[index]
            del sub_type_list[index]
        
        offline_split = np.load('./dataset/{}/client_split_dict.npy'.format(args.dataset), allow_pickle=True).item()
        data_indices = [[] for _ in range(args.num_clients)]
        for data_ind, rea_type in enumerate(sub_type_list):
            for client_ind in range(args.num_clients):
                if rea_type in offline_split[client_ind]:
                    data_indices[client_ind].append(data_ind)
                    break
        print(sum([len(indices) for indices in data_indices]))

        for client_ind in range(args.num_clients):
            client_react_list = [sub_react_list[data_ind] for data_ind in data_indices[client_ind]]
            client_prod_list = [sub_prod_list[data_ind] for data_ind in data_indices[client_ind]]
            savedir = './dataset/TPL_reproduced/client_{}/'.format(client_ind + 9)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            save_dir = os.path.join(savedir, data_set)
            src_data, tgt_data = preprocess(
                save_dir,
                client_react_list,
                client_prod_list,
                data_set,
                args.augmentation,
                reaction_types=None,
                root_aligned=not args.canonical,
                character=args.character,
                processes=args.processes,
            )
