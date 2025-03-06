import os
import random
import shutil

DATASETS = ['mixed']
VERSION = 'v1'
BASE_PATH = './dataset'
CLIENT_COUNT = 4
SPLITS_TO_PROCESS = ['train', 'val']
FILES = ['src', 'tgt']
ERROR_RATIOS = [0.05, 0.1, 0.15, 0.2]

def shuffle_string(tokens):
    tokens = tokens.strip().split()
    random.shuffle(tokens)
    return ' '.join(tokens) + '\n'


def swap_src_tgt_lines(src_file, tgt_file, new_src_file, new_tgt_file, error_ratio):
    with open(src_file, 'r', encoding='utf-8') as src, \
         open(tgt_file, 'r', encoding='utf-8') as tgt:
        src_lines = src.readlines()
        tgt_lines = tgt.readlines()

        if len(src_lines) != len(tgt_lines):
            print(f"Error: {src_file} and {tgt_file} have different numbers of lines.")
            return
        
        line_indices = list(range(len(src_lines)))
        random.shuffle(line_indices)
        swap_count = int(len(src_lines) * error_ratio)
        swap_indices = line_indices[:swap_count]
        
        new_src_lines = []
        new_tgt_lines = []
        for index in range(len(src_lines)):
            if index in swap_indices:
                import pdb; pdb.set_trace()
                new_src_lines.append(shuffle_string(src_lines[index]))
                new_tgt_lines.append(shuffle_string(tgt_lines[index]))
            else:
                new_src_lines.append(src_lines[index])
                new_tgt_lines.append(tgt_lines[index])

        with open(new_src_file, 'w', encoding='utf-8') as f:
            f.writelines(new_src_lines)
        with open(new_tgt_file, 'w', encoding='utf-8') as f:
            f.writelines(new_tgt_lines)


def gen_corrupt_data():
    for dataset in DATASETS:
        for client_id in range(1, CLIENT_COUNT + 1):
            for error_ratio in ERROR_RATIOS:
                new_client_dir_base = os.path.join(
                    BASE_PATH, 
                    dataset,
                    VERSION,
                    f'client_{client_id}_p{int(error_ratio * 100):02d}'
                )
                
                for split in SPLITS_TO_PROCESS:
                    split_dir = os.path.join(BASE_PATH, dataset, f'client_{client_id}', split)
                    new_split_dir = os.path.join(new_client_dir_base, split)
                    
                    os.makedirs(new_split_dir, exist_ok=True)

                    src_file_path = os.path.join(split_dir, f'{FILES[0]}-{split}.txt')
                    tgt_file_path = os.path.join(split_dir, f'{FILES[1]}-{split}.txt')
                    
                    new_src_file_path = os.path.join(new_split_dir, f'{FILES[0]}-{split}.txt')
                    new_tgt_file_path = os.path.join(new_split_dir, f'{FILES[1]}-{split}.txt')
                    
                    swap_src_tgt_lines(src_file_path, tgt_file_path, new_src_file_path, new_tgt_file_path, error_ratio)

                test_dir = os.path.join(BASE_PATH, dataset, f'client_{client_id}', 'test')
                new_test_dir = os.path.join(new_client_dir_base, 'test')
                os.makedirs(new_test_dir, exist_ok=True)
                shutil.copytree(test_dir, new_test_dir, dirs_exist_ok=True)

    print("Data corruption process completed.")

def verify_corruption_ratio(orig_src_file, orig_tgt_file, new_src_file, new_tgt_file, expected_ratio, tolerance=0.01):
    """
    Verify that the actual corruption ratio matches the expected ratio within tolerance.
    
    Args:
        orig_src_file: Path to original source file
        orig_tgt_file: Path to original target file
        new_src_file: Path to corrupted source file 
        new_tgt_file: Path to corrupted target file
        expected_ratio: Expected corruption ratio (0-1)
        tolerance: Acceptable difference between actual and expected ratio
        
    Returns:
        bool: True if verification passes, False otherwise
    """
    with open(orig_src_file, 'r') as f:
        orig_src_lines = f.readlines()
    with open(orig_tgt_file, 'r') as f:
        orig_tgt_lines = f.readlines()
    with open(new_src_file, 'r') as f:
        new_src_lines = f.readlines()
    with open(new_tgt_file, 'r') as f:
        new_tgt_lines = f.readlines()
        
    if len(orig_src_lines) != len(new_src_lines) or len(orig_tgt_lines) != len(new_tgt_lines):
        print(f"Error: Files have different lengths")
        return False
        
    swapped_count = 0
    for i in range(len(orig_src_lines)):
        # if new_src_lines[i] == orig_tgt_lines[i] and new_tgt_lines[i] == orig_src_lines[i]:
        #     swapped_count += 1
        if not new_src_lines[i] == orig_src_lines[i] and not new_tgt_lines[i] == orig_tgt_lines[i]:
            swapped_count += 1
            
    actual_ratio = swapped_count / len(orig_src_lines)
    
    if abs(actual_ratio - expected_ratio) <= tolerance:
        # print(f"Verification passed for {new_src_file}")
        # print(f"Expected ratio: {expected_ratio:.3f}, Actual ratio: {actual_ratio:.3f}")
        return True
    else:
        print(f"Verification failed for {new_src_file}")
        print(f"Expected ratio: {expected_ratio:.3f}, Actual ratio: {actual_ratio:.3f}")
        print(f"Difference exceeds tolerance of {tolerance}")
        return False

def verify_corruption_ratios():
    # Verify corruption ratios for all generated files
    print("\nVerifying corruption ratios...")
    for dataset in DATASETS:
        for client_id in range(1, CLIENT_COUNT + 1):
            for error_ratio in ERROR_RATIOS:
                client_dir = os.path.join(
                    BASE_PATH,
                    dataset,
                    VERSION,
                    f'client_{client_id}_p{int(error_ratio * 100):02d}'
                )
                orig_client_dir = os.path.join(BASE_PATH, dataset, f'client_{client_id}')
                
                for split in SPLITS_TO_PROCESS:
                    orig_src_file = os.path.join(orig_client_dir, split, f'{FILES[0]}-{split}.txt')
                    orig_tgt_file = os.path.join(orig_client_dir, split, f'{FILES[1]}-{split}.txt')
                    new_src_file = os.path.join(client_dir, split, f'{FILES[0]}-{split}.txt')
                    new_tgt_file = os.path.join(client_dir, split, f'{FILES[1]}-{split}.txt')
                    
                    if not verify_corruption_ratio(orig_src_file, orig_tgt_file, new_src_file, new_tgt_file, error_ratio):
                        print(f"Warning: Corruption verification failed for {client_dir}/{split}")

    print("Verification process completed.")


if __name__ == "__main__":
    gen_corrupt_data()
    verify_corruption_ratios()
