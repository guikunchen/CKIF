import csv
import random

random.seed(33)

test_outfp = open('dataset/USPTO_TPL/raw_test.csv', 'w')
test_writer = csv.DictWriter(test_outfp, fieldnames=['id', 'class', 'reactants>reagents>production'])
test_writer.writeheader()
with open('dataset/USPTO_TPL/uspto_1k_TPL_test.tsv', 'r') as fp:
    reader = csv.DictReader(fp, delimiter='\t')
    labels = []
    for item in reader:
        new_item = {'id': item['ID'], 'class': item['labels'], 'reactants>reagents>production': item['reactants'] + '>>' + item['products']}
        for v in new_item.values():
            assert ',' not in v
        test_writer.writerow(new_item)

val_outfp = open('dataset/USPTO_TPL/raw_val.csv', 'w')
train_outfp = open('dataset/USPTO_TPL/raw_train.csv', 'w')
val_writer = csv.DictWriter(val_outfp, fieldnames=['id', 'class', 'reactants>reagents>production'])
train_writer = csv.DictWriter(train_outfp, fieldnames=['id', 'class', 'reactants>reagents>production'])
val_writer.writeheader()
train_writer.writeheader()
with open('dataset/USPTO_TPL/uspto_1k_TPL_train_valid.tsv', 'r') as fp:
    reader = csv.DictReader(fp, delimiter='\t')
    labels = []
    items = [item for item in reader]
    all_indices = [i for i in range(len(items))]
    val_indices = set(random.sample(all_indices, 44511))
    train_indices = set(all_indices) - val_indices
    print(len(all_indices), len(train_indices), len(val_indices))
    for i, item in enumerate(items):
        new_item = {'id': item['ID'], 'class': item['labels'], 'reactants>reagents>production': item['reactants'] + '>>' + item['products']}
        for v in new_item.values():
            assert ',' not in v
        if i in val_indices:
            val_writer.writerow(new_item)
        elif i in train_indices:
            train_writer.writerow(new_item)
        else:
            raise IndexError
