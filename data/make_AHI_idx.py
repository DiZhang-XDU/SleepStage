import os, pickle
import csv

# load all used name
with open('./prepared_data/SHHS-NPZ/refer.txt', 'r') as f:
    lines = f.readlines()
name_dict = {}
for line in lines:
    name_dict[line[52:58]] = line[:4]


# prepare
row_now = -1
name_severity_dict = {}
csv_path = 'D:/SLEEP/shhs/datasets/shhs1-dataset-0.13.0.csv'

# laod csv and go
with open(csv_path, 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        row_now += 1
        if row_now == 0:
            column_name = row.index('nsrrid')
            column_ahi = row.index('ahi_a0h3a')
            continue
        name = row[column_name]
        if name not in name_dict:
            continue
        ahi = float(row[column_ahi])
        severity = 0 if ahi < 5 else 1 if ahi < 30 else 2
        name_severity_dict[name_dict[name]] = severity
    
assert len(name_severity_dict) == len(name_dict)
with open('./prepared_data/eval_ahi.pkl', 'wb') as f:
    pickle.dump(name_severity_dict, f)
    