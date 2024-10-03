import json
import matplotlib.pyplot as plt
import torch
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan']
# name_list = ['tst-rl-gpt2-xl.json', 'tst-dpo-gpt2-xl.json', 'tst-dpo-multi-gpt2-xl.json']
# save_dir = '/hpcgpfs01/scratch/gzhao/rl-prompt/examples/text-style-transfer/outputs/2024-03-22'
# name_list = ['outputs/2024-03-27/15-30-10tst-ipo-gpt2-xl', 
#              'outputs/2024-03-27/15-29-39tst-rl-gpt2-xl', 
#              'outputs/2024-03-27/15-29-00tst-ipo-multi-gpt2-xl'
#              ]
save_dir = '/hpcgpfs01/scratch/gzhao/rl-prompt/examples/few-shot-classification'

def find_pareto_front(samples):
    """
    Find the Pareto front of given sample points.
    
    :param samples: Torch tensor of size N*2 containing sample points (N samples, 2 objectives).
    :return: Pareto front as a torch tensor.
    """
    pareto_front = []
    dominated_by = torch.zeros(samples.size(0), dtype=torch.bool)
    
    for idx, point1 in enumerate(samples):
        if not dominated_by[idx]:
            pareto_front.append(idx)
            for j, point2 in enumerate(samples):
                if idx != j:
                    if torch.all(point1 >= point2):
                        dominated_by[j] = True
                    elif torch.all(point1 <= point2):
                        dominated_by[idx] = True
                        pareto_front.remove(idx)
                        break
    pareto_front_tensor = samples[pareto_front]
    
    return pareto_front_tensor

def calculate_dominating_volume(pareto_front, ref_point):
    """
    Calculate the dominating volume of the Pareto front with respect to a reference point.
    
    :param pareto_front: Torch tensor of size N*2 containing points in the Pareto front (N points, 2 objectives).
    :param ref_point: Torch tensor of size 2 representing the reference point.
    :return: Dominating volume of the Pareto front.
    """
    # Sort Pareto front based on the first objective (ascending order)
    sorted_pareto_front = pareto_front[pareto_front[:, 0].argsort()]
    
    # Initialize dominating volume
    dominating_volume = 0.0
    
    # Initialize the upper left corner of the rectangle
    upper_left_corner = ref_point.clone()
    
    # Iterate through sorted Pareto front
    for point in sorted_pareto_front:
        # Calculate the width and height of the rectangle
        width = point[0] - upper_left_corner[0]  
        height = point[1] - upper_left_corner[1]  
        
        # Update dominating volume by adding the area of the rectangle
        dominating_volume += width * height
        
        # Update the upper left corner for the next rectangle
        upper_left_corner[0] = point[0]
    
    return dominating_volume

import os
date_list = ['09-26', '09-27', '09-28', '09-29', '09-30']
json_files = []
for date in date_list:
    file_path = "/pscratch/sd/g/gzhao27/rl-prompt/examples/few-shot-classification/outputs/2024-"+date
    file_list = os.listdir(file_path)
    json_files += [file_path+"/"+file for file in file_list if file.endswith('.json')]
    

label_list = ('RlPrompt', "HVI", "Reward-Guided DPO", "Reward-Guided IPO", "Dominance-Only DPO", "Dominance-Only IPO", "ParetoPrompt DPO", "ParetoPrompt IPO")
all_files = [[]]*8
all_files[0] = [file for file in json_files if "RlPrompt" in file]
all_files[1] = [file for file in json_files if "HVI" in file]
all_files[2] = [file for file in json_files if "Reward-Guided-DPO" in file]
all_files[3] = [file for file in json_files if "Reward-Guided-IPO" in file]
all_files[4] = [file for file in json_files if "Dominance-Only-DPO" in file]
all_files[5] = [file for file in json_files if "Dominance-Only-IPO" in file]
all_files[6] = [file for file in json_files if "ParetoPrompt-DPO" in file]
all_files[7] = [file for file in json_files if "ParetoPrompt-IPO" in file]


from omegaconf import OmegaConf

# Load the saved config from a YAML file
config = OmegaConf.load("/pscratch/sd/g/gzhao27/rl-prompt/examples/few-shot-classification/config.yaml")
config.base_path = '/pscratch/sd/g/gzhao27/rl-prompt/examples/few-shot-classification/data'
