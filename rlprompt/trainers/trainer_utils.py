import torch
from torch import optim, nn
import numpy as np
import random
from typing import Callable

def get_default_train_op(model: nn.Module,
                         optimizer: optim.Optimizer,
                         gradient_clip: bool,
                         gradient_clip_norm: float) -> Callable[[], None]: 
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=learning_rate)

    def _train_op(): 
        if gradient_clip: 
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        return optimizer.state_dict()

    return _train_op

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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


def evaluate_model_dominate_volume(model, batch, reference_point=torch.tensor([0., 0.]), dominate_evaluate_num=16):
    # dominate_evaluate_num: the num of prompts generated to evaluate the Pareto Front
    (logits, logits_, output_tokens, output_ids, sequence_lengths) = model._decode_sampling(batch=batch, batch_size=dominate_evaluate_num)
    content_list = []
    style_list = []
    for prompt in output_tokens:    
        output_token_list = [prompt]*len(batch['source_texts'])
        sum_reward, content_reward, style_reward, rewards_log= model.compute_rewards(batch=batch, output_tokens=output_token_list, multi_optimize=True)
        content_list.append(content_reward.mean().item())
        style_list.append(style_reward.mean().item())
    
    content = torch.tensor(content_list)
    style = torch.tensor(style_list)
    samples = torch.stack((content, style), dim=1)
    pareto_front_tensor = find_pareto_front(samples)
    
    dominating_volume = calculate_dominating_volume(pareto_front_tensor, reference_point).item()
    
    performance = {}
    performance['dominating_volume'] = dominating_volume
    performance['content'] = content.tolist()
    performance['style'] = style.tolist()
    
    return dominating_volume, performance
