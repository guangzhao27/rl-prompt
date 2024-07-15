import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, Union, List, Tuple
import os
import wandb
import json
import click
import torch.nn.functional as F
import random
from rlprompt.modules import BaseModule
from rlprompt.utils import utils
from .trainer_utils import get_default_train_op, set_random_seed, find_pareto_front, calculate_dominating_volume, evaluate_model_dominate_volume
import copy
from collections import defaultdict


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

class Trainer:
    """Trainer that runs for a specified number of epochs. 

    Each epoch can run for a specified number of batches.
    Evaluation is done at the end of each epoch """

    def __init__(
        self,
        module: BaseModule,
        # Train params
        train_dataset: Optional[Dataset],
        train_batch_size: int,
        train_shuffle: bool,
        train_drop_last: bool,
        num_train_epochs: int,
        max_train_steps: int,
        # Eval params
        do_eval: bool,
        eval_dataset: Optional[Dataset],
        eval_batch_size: int,
        eval_steps: int,
        # Save params
        do_save: bool,
        save_dir: str,
        save_steps: int,
        # Optimizer params
        learning_rate: float,
        gradient_clip: bool,
        gradient_clip_norm: float,
        # Checkpoint params
        checkpoint_path: Optional[str],
        # Random seed
        random_seed: Optional[int],
        # Wandb reporting
        report_to_wandb: bool,
        project_name: Optional[str],
        run_name: Optional[str]
    ):
        assert do_eval == False or eval_dataset is not None, \
            "Need to have eval_dataset if do_eval is True"
        self.module = module

        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.train_drop_last = train_drop_last
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps

        self.do_eval = do_eval
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps

        self.do_save = do_save
        self.save_dir = save_dir
        self.save_steps = save_steps
        
        self.optimizer = torch.optim.Adam(self.module._model.parameters(),
                           lr=learning_rate)

        self.train_op = get_default_train_op(self.module._model,
                                             learning_rate,
                                             gradient_clip,
                                             gradient_clip_norm)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        if random_seed is not None:
            set_random_seed(random_seed)

        self.report_to_wandb = report_to_wandb
        self.project_name = project_name
        self.run_name = run_name

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.module.load_state_dict(checkpoint["model_state_dict"])
        print(click.style(f"Loaded module from {checkpoint_path}", fg="green"))

    def _get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          shuffle=self.train_shuffle,
                          batch_size=self.train_batch_size,
                          drop_last=self.train_drop_last)

    # @torch.no_grad
    def _train_step(
        self,
        step: int,
        batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        model = self.module.train()
        model._pre_steps(step)

        loss, batch_log = model(batch)
        loss.backward()
        batch_log['SQL_ON/rewards/loss'] = loss.item()
        self.optimizer.step()
        self.optimizer.zero_grad()
        optimizer_state = self.optimizer.state_dict()
        # optimize_state = self.train_op()
        
        # if self.report_to_wandb:
        #     wandb.log(batch_log, step=step)

        return batch_log, optimizer_state

    def train(self,
              report_to_wandb: Optional[bool] = None,
              project_name: Optional[str] = None,
              run_name: Optional[str] = None,
              config: Optional["DictConfig"] = None) -> None:
        # Configure Wandb reporting
        if report_to_wandb is None:
            report_to_wandb = self.report_to_wandb
        if project_name is None:
            project_name = self.project_name
        if run_name is None: 
            run_name = self.run_name
        # if config is not None: 
        #     config = eval(str(config))
        if report_to_wandb:
            os.environ["WANDB_API_KEY"] = "e48870a815eed7ccfabbb6d1a0e40f9a618bfa88"
            wandb.init(project=project_name, name=run_name, config=eval(str(config)))
            wandb.watch(self.module, log=None)
            print(wandb.run.dir)
            print(self.run_name)

        # Create saving path
        eval_save_dir = os.path.join(self.save_dir, "eval")
        ckpt_save_dir = os.path.join(self.save_dir, "ckpt")
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        train_dataloader = self._get_train_dataloader()
        # Determine whether to train by epoch or steps
        if self.max_train_steps < 0:
            total_train_epochs = self.num_train_epochs
        else:
            num_batches_per_epoch = len(train_dataloader)
            total_train_epochs = \
                (self.max_train_steps // num_batches_per_epoch
                 + int(self.max_train_steps % num_batches_per_epoch > 0))

        # Determine whether to evaluate by epoch or steps
        eval_by_steps = self.eval_steps > 0
        # Determine whether to save by epoch or steps
        save_by_steps = self.save_steps > 0

        total_steps = 0
        for epoch in range(total_train_epochs):
            for step, batch in enumerate(train_dataloader):
                total_steps += 1
                if config.model_path and step < config.load_step:
                    continue
                batch_log, _ = self._train_step(total_steps, batch)
                if report_to_wandb:
                    wandb.log(batch_log, step=total_steps) # The loss is calculated by averaging the loss of generated prompts from the prmopt modle 
                

                if self.do_eval and eval_by_steps \
                        and total_steps % self.eval_steps == 0:
                    print('start evaluation')
                    output_save_path = \
                        os.path.join(eval_save_dir,
                                     f'outputs.step.{total_steps}.json')
                    eval_log, performance = self.evaluate(output_save_path=output_save_path, dominate_evaluate_num=config.dominate_evaluate_num)
                    if report_to_wandb:
                        wandb.log(eval_log, step=total_steps)
                    
                    file_name = run_name + '.json'
                    if os.path.exists(file_name):
                        with open(file_name, 'r') as f:
                            performance_dict = json.load(f)
                    else:
                        performance_dict = {}
                    performance_dict[total_steps] = performance
                    with open(file_name, 'w') as f:
                        json.dump(performance_dict, f)

                if self.do_save and save_by_steps \
                        and total_steps % self.save_steps == 0:
                    torch.save({"steps": total_steps,
                                "model_state_dict": self.module.state_dict(), 
                                "optimizer_state": self.optimizer.state_dict(), 
                                },
                               os.path.join(ckpt_save_dir,
                                            f"ckpt.step.{total_steps}.pth"))

                if total_steps == self.max_train_steps:
                    break

            if self.do_eval and not eval_by_steps:
                output_save_path = os.path.join(eval_save_dir,
                                                f'outputs.epoch.{epoch+1}.json')
                eval_log, performance = self.evaluate(output_save_path=output_save_path)
                wandb.log(eval_log, step=total_steps)

            if self.do_save and not save_by_steps:
                torch.save({"steps": total_steps,
                            "model_state_dict": self.module.state_dict()},
                           os.path.join(ckpt_save_dir,
                                        f"ckpt.epoch.{epoch+1}.pth"))

    def _get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
        return DataLoader(eval_dataset,
                          batch_size=self.eval_batch_size)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        output_save_path: Optional[str] = None,
        dominate_evaluate_num: int=16,
        compute_scores: bool = True
    ) -> Dict[str, np.number]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_dataloader = self._get_eval_dataloader(eval_dataset)

        model = self.module.eval()
        hypos = []
        scores: List[List[str]] = []
        for batch in eval_dataloader:
            infer_outputs: Dict[str, Union[torch.Tensor, List[List[str]]]]
            infer_outputs = model.infer(batch)
            hypos += infer_outputs['sample_tokens'] #hypos are the most likely prompts

            score, content_reward, style_reward, score_log = model.compute_rewards(
                batch=batch,
                output_tokens=infer_outputs['sample_tokens'], multi_optimize=True)
            scores += score.detach().tolist()
        
            # generate many prompt like load and log the mean style and content and calculate the dominating area
            dominate_volume, performance = evaluate_model_dominate_volume(model=model, batch=batch, dominate_evaluate_num=dominate_evaluate_num)
        if output_save_path is not None:
            json.dump({'output_tokens': hypos,
                       'scores': scores},
                      open(output_save_path, 'w'))

        score = score.mean().item()
        content_reward = content_reward.mean().item()
        style_reward = style_reward.mean().item()

        utils.add_prefix_to_dict_keys_inplace(
            score_log,
            prefix=f"eval/rewards/")

        print('Finish Eval')
        return utils.unionize_dicts([
            score_log,
            # gem_scores_dict,
            {
                f"eval/dominate_volume": dominate_volume,
                f"eval/score": score,
                f"eval/content":content_reward, 
                f"eval/style": style_reward, 
                f"eval/output_length": np.mean([len(tokens) \
                                                for tokens in hypos])
            }
        ]), performance
        
    def load_pretrain(self, ckpt_path, training_device):
        # check point keys ['steps', 'model_state_dict', 'optimizer_state']
        if training_device == "cpu":
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        elif training_device == "cuda":
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cuda"))
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.module.load_state_dict(checkpoint['model_state_dict'])
        
class DPO_Trainer(Trainer):
    def __init__(
        self,
        module: BaseModule,
        # Train params
        train_dataset: Optional[Dataset],
        train_batch_size: int,
        train_shuffle: bool,
        train_drop_last: bool,
        num_train_epochs: int,
        max_train_steps: int,
        # Eval params
        do_eval: bool,
        eval_dataset: Optional[Dataset],
        eval_batch_size: int,
        eval_steps: int,
        # Save params
        do_save: bool,
        save_dir: str,
        save_steps: int,
        # Optimizer params
        learning_rate: float,
        gradient_clip: bool,
        gradient_clip_norm: float,
        # Checkpoint params
        checkpoint_path: Optional[str],
        # Random seed
        random_seed: Optional[int],
        # Wandb reporting
        report_to_wandb: bool,
        project_name: Optional[str],
        run_name: Optional[str],
        dpo_loss_config: Dict[str, Any],
    ):
        super().__init__(
            module,
            train_dataset,
            train_batch_size,
            train_shuffle,
            train_drop_last,
            num_train_epochs,
            max_train_steps,
            do_eval,
            eval_dataset,
            eval_batch_size,
            eval_steps,
            do_save,
            save_dir,
            save_steps,
            learning_rate,
            gradient_clip,
            gradient_clip_norm,
            checkpoint_path,
            random_seed,
            report_to_wandb,
            project_name,
            run_name, 
        )
        self.dpo_loss_config = dpo_loss_config
        
    def _train_step(self, step: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        policy_model = self.module.train()
        policy_model._pre_steps(step)  
        ## it updates reference model, while with learning rate=0.0, it doesn't learn anything
        # /hpcgpfs01/scratch/gzhao/rl-prompt/rlprompt/modules/sql_module.py _pre_steps()
        
        
        
        
        
        (policy_logits, ref_logits, output_tokens, output_ids, sequence_lengths) = \
            policy_model._decode_sampling(batch=batch)  ## TODO: where should these samples being sampled from??
            
        raw_rewards, rewards_log = \
            policy_model.compute_rewards(batch=batch, 
                                output_tokens=output_tokens,
                                mode="train")
        
        reward_matrix = policy_model._reward_shaping_func(raw_rewards)
        sample_num = reward_matrix.shape[0]
        token_lenght = output_ids.shape[1]
        
        policy_all_logps = policy_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        
        # ref_model = copy.deepcopy(policy_model).eval()
        # ref_model._pre_steps(step)
        # outputs_ = ref_model.teacher_forcing(**batch)  ## what is ** batch, check if ref_model update, follow the debuuging process s
        
        # ref_logits = outputs_['sample_logits']
        if self.dpo_loss_config.name == 'reinforce':
            log_prob = F.log_softmax(policy_logits, dim=-1)
            policy_all_logps = log_prob[
                torch.arange(sample_num).unsqueeze(1), 
                torch.arange(token_lenght), 
                output_ids
            ].sum(axis=1)
            losses = -policy_all_logps*raw_rewards
        else:
            ref_all_logps = ref_logits[
                torch.arange(sample_num).unsqueeze(1), 
                torch.arange(token_lenght), 
                output_ids
            ].sum(axis=1)
            
            
            chosen_idx_list = torch.zeros(sample_num // 2, dtype=torch.long)
            rejected_idx_list = torch.zeros(sample_num // 2, dtype=torch.long)
            for i in range(sample_num // 2):
                if reward_matrix[2*i] > reward_matrix[2*i + 1]:
                    chosen_idx_list[i] = 2 * i
                    rejected_idx_list[i] = 2 * i + 1
                else:
                    chosen_idx_list[i] = 2 * i + 1
                    rejected_idx_list[i] = 2 * i## make it random?
            
            
            
            policy_chosen_logps = policy_all_logps[chosen_idx_list]
            policy_rejected_logps = policy_all_logps[rejected_idx_list]
            with torch.no_grad():
                reference_chosen_logps = ref_all_logps[chosen_idx_list]
                reference_rejected_logps = ref_all_logps[rejected_idx_list]
            
            if self.dpo_loss_config.name == 'dpo':
                    loss_kwargs = {'beta': self.dpo_loss_config.beta, 'reference_free': self.dpo_loss_config.reference_free, 'label_smoothing': self.dpo_loss_config.label_smoothing, 'ipo': False}
            elif self.dpo_loss_config.name == 'ipo':
                loss_kwargs = {'beta': self.dpo_loss_config.beta, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {self.dpo_loss_config.name}')

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        loss = losses.mean()
        loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        optimizer_state = self.optimizer.state_dict()
        
        # optimizer_state = self.train_op()
        
        batch_log = rewards_log  ## add more recorded metrics in this log
        
        utils.add_prefix_to_dict_keys_inplace(
                rewards_log, prefix="SQL_ON/rewards/")
        
        batch_log = rewards_log
        batch_log['dpo_loss'] = loss

        return batch_log, optimizer_state
       
class DPO_O2_Trainer(Trainer):
    def __init__(
        self,
        module: BaseModule,
        # Train params
        train_dataset: Optional[Dataset],
        train_batch_size: int,
        train_shuffle: bool,
        train_drop_last: bool,
        num_train_epochs: int,
        max_train_steps: int,
        # Eval params
        do_eval: bool,
        eval_dataset: Optional[Dataset],
        eval_batch_size: int,
        eval_steps: int,
        # Save params
        do_save: bool,
        save_dir: str,
        save_steps: int,
        # Optimizer params
        learning_rate: float,
        gradient_clip: bool,
        gradient_clip_norm: float,
        # Checkpoint params
        checkpoint_path: Optional[str],
        # Random seed
        random_seed: Optional[int],
        # Wandb reporting
        report_to_wandb: bool,
        project_name: Optional[str],
        run_name: Optional[str],
        dpo_loss_config: Dict[str, Any],
    ):
        super().__init__(
            module,
            train_dataset,
            train_batch_size,
            train_shuffle,
            train_drop_last,
            num_train_epochs,
            max_train_steps,
            do_eval,
            eval_dataset,
            eval_batch_size,
            eval_steps,
            do_save,
            save_dir,
            save_steps,
            learning_rate,
            gradient_clip,
            gradient_clip_norm,
            checkpoint_path,
            random_seed,
            report_to_wandb,
            project_name,
            run_name, 
        )
        self.dpo_loss_config = dpo_loss_config
        
    def loss_func(self, step, batch, config):
        policy_model = self.module.train()
        policy_model._pre_steps(step)  
        
        (policy_logits, ref_logits, output_tokens, output_ids, sequence_lengths) = \
            policy_model._decode_sampling(batch=batch) # for multiobjective batch size should be large
        
        sample_num = policy_logits.shape[0]
        token_lenght = policy_logits.shape[1]
        
        policy_all_logps = policy_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        
        ref_all_logps = ref_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        return policy_all_logps, ref_all_logps
        

    def _train_step(self, step: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        # policy_model = self.module.train()
        # policy_model._pre_steps(step)  
        
        # (policy_logits, ref_logits, output_tokens, output_ids, sequence_lengths) = \
        #     policy_model._decode_sampling(batch=batch) # for multiobjective batch size should be large
        
        # sample_num = policy_logits.shape[0]
        # token_lenght = policy_logits.shape[1]
        
        # policy_all_logps = policy_logits[
        #     torch.arange(sample_num).unsqueeze(1), 
        #     torch.arange(token_lenght), 
        #     output_ids
        # ].sum(axis=1)
        
        # ref_all_logps = ref_logits[
        #     torch.arange(sample_num).unsqueeze(1), 
        #     torch.arange(token_lenght), 
        #     output_ids
        # ].sum(axis=1)
        
        # #bypass the compute_rewards function: /hpcgpfs01/scratch/gzhao/rl-prompt/rlprompt/modules/sql_module.py
        # sum_reward, content_reward, style_reward, rewards_log= \
        #     policy_model.compute_rewards(batch=batch, 
        #                         output_tokens=output_tokens,
        #                         mode="train", multi_optimize=self.dpo_loss_config.multi_optimize)
        
        # # add non dominate punishment here
        # if self.dpo_loss_config.nondominate_punishment is None:
        #     # compare pairs of data and find dominate pairs
        #     dominate_dict = self.find_dominate_indices(content_reward, style_reward)
            
        #     if not bool(dominate_dict) and self.dpo_loss_config.nondominate_punishment is None:
        #         utils.add_prefix_to_dict_keys_inplace(
        #             rewards_log, prefix="SQL_ON/rewards/")
        #         batch_log = rewards_log
        #         return batch_log, None
            
        #     chosen_idx_list = list(dominate_dict.keys()) 
        #     rejected_idx_list = []
        #     for i in chosen_idx_list:
        #         j = random.choice(dominate_dict[i])
        #         rejected_idx_list.append(j)
            
            
        #     policy_chosen_logps = policy_all_logps[chosen_idx_list]
        #     policy_rejected_logps = policy_all_logps[rejected_idx_list]
        #     with torch.no_grad():
        #         reference_chosen_logps = ref_all_logps[chosen_idx_list]
        #         reference_rejected_logps = ref_all_logps[rejected_idx_list]
                
        #     # different dpo loss
        #     if self.dpo_loss_config.name == 'dpo':
        #             loss_kwargs = {'beta': self.dpo_loss_config.beta, 'reference_free': self.dpo_loss_config.reference_free, 'label_smoothing': self.dpo_loss_config.label_smoothing, 'ipo': False}
        #     elif self.dpo_loss_config.name == 'ipo':
        #         loss_kwargs = {'beta': self.dpo_loss_config.beta, 'ipo': True}
        #     else:
        #         raise ValueError(f'unknown loss {self.dpo_loss_config.name}')

        #     losses, chosen_rewards, rejected_rewards = preference_loss(
        #         policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
        
        # elif self.dpo_loss_config.nondominate_punishment == 'prob_diff':
            
        #     # randomly split data into sample_num//2 pairs
        #     idxs = list(range(sample_num))
        #     random.shuffle(idxs)
        #     idx_pairs = [(idxs[i], idxs[i+1]) for i in range(0, sample_num, 2)]
        #     idx_pairs = [(i, j) for i in range(sample_num) for j in range(i+1, sample_num)]
            
        #     chosen_idx_list = []
        #     rejected_idx_list = []
        #     non_dominate_list = []
        #     for idx1, idx2 in idx_pairs:
        #         c1 = content_reward[idx1]
        #         c2 = content_reward[idx2]
        #         s1, s2 = style_reward[idx1], style_reward[idx2]
                
        #         if c1>=c2 and s1>=s2 and (c1>c2 or s1>s2):
        #             chosen_idx_list.append(idx1)
        #             rejected_idx_list.append(idx2)
        #         elif c1<=c2 and s1<=s2 and (c1<c2 or s1<s2):
        #             chosen_idx_list.append(idx2)
        #             rejected_idx_list.append(idx1)
        #         else:
        #             non_dominate_list.append((idx1, idx2))
        #     losses1 = torch.tensor([]).to(policy_all_logps.device)
        #     losses2 = torch.tensor([]).to(policy_all_logps.device)
            
        #     if non_dominate_list:
        #         logit1 = policy_all_logps[[x[0] for x in non_dominate_list]]
        #         logit2 = policy_all_logps[[x[1] for x in non_dominate_list]]
        #         epsilon = self.dpo_loss_config.epsilon  # Epsilon value
        #         loss_fn = EpsilonInsensitiveLoss(epsilon)
        #         losses1 = loss_fn(logit1, logit2)
                
        #     if chosen_idx_list:
        #         policy_chosen_logps = policy_all_logps[chosen_idx_list]
        #         policy_rejected_logps = policy_all_logps[rejected_idx_list]
        #         with torch.no_grad():
        #             reference_chosen_logps = ref_all_logps[chosen_idx_list]
        #             reference_rejected_logps = ref_all_logps[rejected_idx_list]
                    
        #         # different dpo loss
        #         if self.dpo_loss_config.name == 'dpo':
        #                 loss_kwargs = {'beta': self.dpo_loss_config.beta, 'reference_free': self.dpo_loss_config.reference_free, 'label_smoothing': self.dpo_loss_config.label_smoothing, 'ipo': False}
        #         elif self.dpo_loss_config.name == 'ipo':
        #             loss_kwargs = {'beta': self.dpo_loss_config.beta, 'ipo': True}
        #         else:
        #             raise ValueError(f'unknown loss {self.dpo_loss_config.name}')

        #         losses2, chosen_rewards, rejected_rewards = preference_loss(
        #             policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
        #     losses = torch.concatenate((losses1, losses2))
        
        # loss = losses.mean()
        # # print(loss)
        
        
        loss, rewards_log = self.dpo_loss_calculation(step, batch, self.dpo_loss_config)
        
        loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        optimizer_state = self.optimizer.state_dict()
        # optimizer_state = self.train_op()
        
        # batch_log = rewards_log  ## add more recorded metrics in this log
        utils.add_prefix_to_dict_keys_inplace(
                rewards_log, prefix="SQL_ON/rewards/")
        
        batch_log = rewards_log
        batch_log['dpo_loss'] = loss

        return batch_log, optimizer_state
    
    def dpo_loss_calculation(self, step, batch, dpo_loss_config):
        policy_model = self.module.train()
        policy_model._pre_steps(step)  
        
        # (policy_logits, ref_logits, output_tokens, output_ids, sequence_lengths) = \
        #     policy_model._decode_sampling(batch=batch) # for multiobjective batch size should be large
        
        # sample_num = policy_logits.shape[0]
        # token_lenght = policy_logits.shape[1]
        
        # policy_all_logps = policy_logits[
        #     torch.arange(sample_num).unsqueeze(1), 
        #     torch.arange(token_lenght), 
        #     output_ids
        # ].sum(axis=1)
        
        # ref_all_logps = ref_logits[
        #     torch.arange(sample_num).unsqueeze(1), 
        #     torch.arange(token_lenght), 
        #     output_ids
        # ].sum(axis=1)
        
        # #bypass the compute_rewards function: /hpcgpfs01/scratch/gzhao/rl-prompt/rlprompt/modules/sql_module.py
        # sum_reward, content_reward, style_reward, rewards_log= \
        #     policy_model.compute_rewards(batch=batch, 
        #                         output_tokens=output_tokens,
        #                         mode="train", multi_optimize=dpo_loss_config.multi_optimize)
        
        sum_reward, content_reward, style_reward, rewards_log, policy_all_logps, ref_all_logps = \
        self.sample_prompt_calculate_logp(batch, policy_model)
        
        sample_num = len(content_reward)
        
        
        idxs = list(range(sample_num))
        random.shuffle(idxs)
        idx_pairs = [(idxs[i], idxs[i+1]) for i in range(0, sample_num, 2)]
        idx_pairs = [(i, j) for i in range(sample_num) for j in range(i+1, sample_num)]
        
        chosen_idx_list = []
        rejected_idx_list = []
        non_dominate_list = []
        for idx1, idx2 in idx_pairs:
            c1 = content_reward[idx1]
            c2 = content_reward[idx2]
            s1, s2 = style_reward[idx1], style_reward[idx2]
            
            if c1>=c2 and s1>=s2 and (c1>c2 or s1>s2):
                chosen_idx_list.append(idx1)
                rejected_idx_list.append(idx2)
            elif c1<=c2 and s1<=s2 and (c1<c2 or s1<s2):
                chosen_idx_list.append(idx2)
                rejected_idx_list.append(idx1)
            else:
                non_dominate_list.append((idx1, idx2))
        losses1 = torch.tensor([]).to(policy_all_logps.device)
        losses2 = torch.tensor([]).to(policy_all_logps.device)
        
        if chosen_idx_list:
            policy_chosen_logps = policy_all_logps[chosen_idx_list]
            policy_rejected_logps = policy_all_logps[rejected_idx_list]
            with torch.no_grad():
                reference_chosen_logps = ref_all_logps[chosen_idx_list]
                reference_rejected_logps = ref_all_logps[rejected_idx_list]
                
            # different dpo loss
            if self.dpo_loss_config.name == 'dpo':
                    loss_kwargs = {'beta': self.dpo_loss_config.beta, 'reference_free': self.dpo_loss_config.reference_free, 'label_smoothing': self.dpo_loss_config.label_smoothing, 'ipo': False}
            elif self.dpo_loss_config.name == 'ipo':
                loss_kwargs = {'beta': self.dpo_loss_config.beta, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {self.dpo_loss_config.name}')

            losses1, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
        
        
        if dpo_loss_config.nondominate_punishment is None:
            pass
        elif dpo_loss_config.nondominate_punishment == 'prob_diff':
            
            if non_dominate_list:
                logit1 = policy_all_logps[[x[0] for x in non_dominate_list]]
                logit2 = policy_all_logps[[x[1] for x in non_dominate_list]]
                epsilon = self.dpo_loss_config.epsilon  # Epsilon value
                loss_fn = EpsilonInsensitiveLoss(epsilon)
                losses2 = loss_fn(logit1, logit2)
                
        losses = torch.concatenate((losses1, losses2))
        
        if len(losses) == 0:
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            loss = losses.mean()
        
        return loss, rewards_log
    
    def sample_prompt_calculate_logp(self, batch, policy_model):
        (policy_logits, ref_logits, output_tokens, output_ids, sequence_lengths) = \
            policy_model._decode_sampling(batch=batch) # for multiobjective batch size should be large
        
        sample_num = policy_logits.shape[0]
        token_lenght = policy_logits.shape[1]
        
        policy_all_logps = policy_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        
        ref_all_logps = ref_logits[
            torch.arange(sample_num).unsqueeze(1), 
            torch.arange(token_lenght), 
            output_ids
        ].sum(axis=1)
        
        #bypass the compute_rewards function: /hpcgpfs01/scratch/gzhao/rl-prompt/rlprompt/modules/sql_module.py
        sum_reward, content_reward, style_reward, rewards_log= \
            policy_model.compute_rewards(batch=batch, 
                                output_tokens=output_tokens,
                                mode="train", multi_optimize=self.dpo_loss_config.multi_optimize)
        
        return sum_reward, content_reward, style_reward, rewards_log, policy_all_logps, ref_all_logps

    
    def pareto_front(self, reward1, reward2):
        combined_rewards = torch.stack([reward1, reward2], dim=1)
    
        # Sort the combined rewards in descending order
        sorted_rewards, indices = torch.sort(combined_rewards, dim=0, descending=True)
        
        # Initialize the Pareto front and other layers indices
        pareto_front_indices = [indices[0, 0]]
        other_layers_indices = []
        
        # Iterate through the sorted rewards to identify the Pareto front and other layers
        for i in range(1, len(sorted_rewards)):
            if sorted_rewards[i, 1] < sorted_rewards[i-1, 1]:
                pareto_front_indices.append(indices[i, 0])
            else:
                other_layers_indices.append(indices[i, 0])
        
        return pareto_front_indices, other_layers_indices
    
    def find_dominate_indices(self, reward1, reward2):
        """
        Find the dominated indices from two reward tensors and return a dictionary.
        
        Parameters:
            reward1 (list or torch.Tensor): Reward tensor 1.
            reward2 (list or torch.Tensor): Reward tensor 2.
            
        Returns:
            dict: A dictionary where keys are indices of dominating elements and values are lists 
                of indices dominated by the corresponding key.
        """
        dominate_dict = {}
        for i, (r1, r2) in enumerate(zip(reward1, reward2)):
            for j, (rr1, rr2) in enumerate(zip(reward1, reward2)):
                if i != j:  # Avoid comparing an element with itself
                    if r1 >= rr1 and r2 >= rr2 and (r1 > rr1 or r2 > rr2):
                        # Element i dominates element j
                        dominate_dict.setdefault(i, []).append(j)
        return dominate_dict
        # self._target_model is the reference model
        #  outputs_ = self._target_model.teacher_forcing(**batch_) this one gives the logtis at: outputs_['sample_logits'].contiguous(),
            #    https://vscode.dev/github/mingkaid/rl-prompt/blob/main/rlprompt/modules/sql_module.py#L216
            
        # policy_chosen_logps = policy_logits[chosen_idx_list]
        # policy_rejected_logps = policy_logits[rejected_idx_list]
        # reference_chosen_logps = ref_logits[chosen_idx_list]
        # reference_rejected_logps = ref_logits[rejected_idx_list]
            
class EpsilonInsensitiveLoss(torch.nn.Module):
    def __init__(self, epsilon):
        super(EpsilonInsensitiveLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        loss = torch.max(torch.abs(y_true - y_pred) - self.epsilon, torch.tensor(0.0))
        return loss
        
        
        