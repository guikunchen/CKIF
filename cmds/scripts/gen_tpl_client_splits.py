import csv
import random
from collections import Counter
import numpy as np

random.seed(21)

NUM_CLIENT = 8
cli2cls = {i: [] for i in range(NUM_CLIENT)}
with open('dataset/USPTO_TPL/raw_train.csv', 'r') as fp:
    reader = csv.DictReader(fp, delimiter=',')
    labels = []
    for item in reader:
        labels.append(int(item['class']))
counter = Counter(labels)
sorted_counter = sorted(list(counter.items()), key=lambda x: -x[-1])
sorted_counter = sorted_counter[240:1000]  # Monitor different distributions. Top 240 classes have too much samples.


for cls_i in range(NUM_CLIENT):
    cur_all_clses = [item[0] for item in sorted_counter[cls_i*len(sorted_counter)//NUM_CLIENT:(cls_i+1)*len(sorted_counter)//NUM_CLIENT]] if cls_i != NUM_CLIENT - 1 else [item[0] for item in sorted_counter[cls_i*len(sorted_counter)//NUM_CLIENT:]]
    left_clses = set(cur_all_clses)
    for cli_i in range(NUM_CLIENT):
        sampled_clses = random.sample(left_clses, len(cur_all_clses) // NUM_CLIENT)
        left_clses = left_clses - set(sampled_clses)
        cli2cls[cli_i].extend(sampled_clses)


cnt = 0
# print(cli2cls)
for cli_i in range(NUM_CLIENT):
    cnt += sum([counter[i] for i in cli2cls[cli_i]])
    print(cli_i, sum([counter[i] for i in cli2cls[cli_i]]))
print(cnt, sum(counter.values()))
np.save('dataset/USPTO_TPL/client_split_dict.npy', cli2cls)
