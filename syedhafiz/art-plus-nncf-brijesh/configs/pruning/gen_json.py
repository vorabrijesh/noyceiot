#!/usr/bin/python3

import json
import sys
import os

def print_usage():
    print('Usage: <base_json_path> <model_name> <pretrained - 0/1> <sample_size - x,x,x,x> <batch_size> <epochs> <prun_target> <filter_importance> <out_json>')

def main(argv):

    if(len(sys.argv) < 9):
        print_usage()
        exit()

    json_path = str(sys.argv[1]) # path to base json file
    model_name = str(sys.argv[2]) # Model name
    pretrained = int(sys.argv[3]) # 0 - False, 1 - True
    sample_size = str(sys.argv[4]) # 1,32,32,3 format
    batch_size = int(sys.argv[5]) # Batch size
    n_epochs = int(sys.argv[6]) # Epochs
    prun_target = float(sys.argv[7]) # Pruning target - [0,1]
    filt_imp = str(sys.argv[8]) # L1, L2, geometric_median
    #prun_steps = int(sys.argv[9])
    out_json = str(sys.argv[9]) # output json file to store changes

    # Error checking args

    if pretrained == 0:
        pretrain = False
    elif pretrained == 1:
        pretrain = True
    else:
        print("Error: pretrained = 0/1")
        exit()

    sample = sample_size.split(',')
    if len(sample) != 4:
        print("Error: sample_size must be 4 digits in form: x,x,x,x")
        exit()

    sample = list(map(int, sample))

    if prun_target > 1 or prun_target < 0:
        print("Error: pruning target must be in range [0,1]")
        exit()

    list_of_filt_imp = ["L1", "L2", "geometric_median"]
    if filt_imp not in list_of_filt_imp:
        print("Error: filt importance must be ", list_of_filt_imp)
        exit()
    
    if not out_json.endswith('.json'):
        print("Error: out_json file must be a .json")
        exit()

    # Read base json
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Modify config file
    data['model'] = model_name
    data['pretrained'] = pretrain
    data['input_info']['sample_size'] = sample
    data['batch_size'] = batch_size
    data['epochs'] = n_epochs
    data['compression']['params']['pruning_target'] = prun_target
    data['compression']['params']['filter_importance'] = filt_imp
    # data['compression']['params']['pruning_steps'] = prun_steps

    # write to new json
    with open(out_json, 'w') as out_file:
        json.dump(data, out_file, indent=4)

if __name__ == "__main__":
   main(sys.argv[1:])
