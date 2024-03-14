import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, Union, List, Tuple
import os
import wandb
import json
import click
import torch.nn.functional as F

from rlprompt.modules import BaseModule
from rlprompt.utils import utils
from .trainer_utils import get_default_train_op, set_random_seed
import copy


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

        loss, batch_log = model(batch)  ## here the loss is obtained, replace this loss with DPO, batch_log can be any dict
        loss.backward()
        self.train_op()

        return batch_log

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
        if config is not None: 
            config = eval(str(config))
        if report_to_wandb:
            os.environ["WANDB_API_KEY"] = "e48870a815eed7ccfabbb6d1a0e40f9a618bfa88"
            wandb.init(project=project_name, name=run_name, config=config)
            wandb.watch(self.module, log=None)

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
                batch_log = self._train_step(step, batch)
                if report_to_wandb:
                    wandb.log(batch_log) # The loss is calculated by averaging the loss of generated prompts from the prmopt modle 
                total_steps += 1

                if self.do_eval and eval_by_steps \
                        and total_steps % self.eval_steps == 0:
                    output_save_path = \
                        os.path.join(eval_save_dir,
                                     f'outputs.step.{total_steps}.json')
                    eval_log = self.evaluate(output_save_path=output_save_path)
                    if report_to_wandb:
                        wandb.log(eval_log)

                if self.do_save and save_by_steps \
                        and total_steps % self.save_steps == 0:
                    torch.save({"steps": total_steps,
                                "model_state_dict": self.module.state_dict()},
                               os.path.join(ckpt_save_dir,
                                            f"ckpt.step.{total_steps}.pth"))

                if total_steps == self.max_train_steps:
                    break

            if self.do_eval and not eval_by_steps:
                output_save_path = os.path.join(eval_save_dir,
                                                f'outputs.epoch.{epoch+1}.json')
                eval_log = self.evaluate(output_save_path=output_save_path)
                wandb.log(eval_log)

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
            hypos += infer_outputs['sample_tokens']

            score, score_log = model.compute_rewards(
                batch=batch,
                output_tokens=infer_outputs['sample_tokens'])
            scores += score.detach().tolist()

        if output_save_path is not None:
            json.dump({'output_tokens': hypos,
                       'scores': scores},
                      open(output_save_path, 'w'))

        score = score.mean().item()

        utils.add_prefix_to_dict_keys_inplace(
            score_log,
            prefix=f"eval/rewards/")

        print('Finish Eval')
        return utils.unionize_dicts([
            score_log,
            # gem_scores_dict,
            {
                f"eval/score": score,
                f"eval/output_length": np.mean([len(tokens) \
                                                for tokens in hypos])
            }
        ])
        
        
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
        self.reference_model = copy.deepcopy(module)
        self.dpo_loss_config = dpo_loss_config
        
    def _train_step(self, step: int, batch: Dict[str, Any]) -> Dict[str, Any]:

        # loss_config.name = "ipo"
        # loss_config.beta = 0.5
        # loss_config.reference_free = False
        # loss_config.label_smoothing = 0.0
        # loss_config.ipo = False
        
        loss_config = self.dpo_loss_config
        policy_model = self.module.train()
        policy_model._target_learning_rate = loss_config.reference_learning_rate
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
        
        if loss_config.name == 'dpo':
                loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
        elif loss_config.name == 'ipo':
            loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
        else:
            raise ValueError(f'unknown loss {loss_config.name}')

        losses, chosen_rewards, rejected_rewards = preference_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        loss = losses.mean()
        
        
        
        loss.backward()
        self.train_op()
        
        batch_log = rewards_log  ## add more recorded metrics in this log
        
        utils.add_prefix_to_dict_keys_inplace(
                rewards_log, prefix="SQL_ON/rewards/")
        
        batch_log = rewards_log
        batch_log['dpo_loss'] = loss

        return batch_log
        
        
        
        # self._target_model is the reference model
        #  outputs_ = self._target_model.teacher_forcing(**batch_) this one gives the logtis at: outputs_['sample_logits'].contiguous(),
            #    https://vscode.dev/github/mingkaid/rl-prompt/blob/main/rlprompt/modules/sql_module.py#L216
            
        # policy_chosen_logps = policy_logits[chosen_idx_list]
        # policy_rejected_logps = policy_logits[rejected_idx_list]
        # reference_chosen_logps = ref_logits[chosen_idx_list]
        # reference_rejected_logps = ref_logits[rejected_idx_list]
            
            
        
        
        # reward_matrix = model.compute_reward_matrix(batch)
        

        # loss.backward()
        
        # return rewards_log
    
        

    # def _dpo_train_step(
    #     self,
    #     step: int, 
    #     batch: Dict[str, Any]
    #     ) -> Dict[str, Any]: 
    #     model = self.module.train()
        
    #     # return a reward matrix for each iteration, 
        
    #     #then calculate the loss to update the policy model, while keep the reference model for several iterations, 
        
    #     # make the code flexible, we can update the reference model, while update the policy model in each iteration, do not follow all their updates
        
    #     loss, batch_log = model(batch)
    #     loss.backward()
    #     self.train_op()
        
    #     return batch_log
# with open("/hpcgpfs01/scratch/gzhao/debug_data.pkl", "wb") as f:
#     debug_data = {
#         "logits": logits,
#         "logits_": logits_,
#         "output_tokens": output_tokens,
#         "output_ids": output_ids, 
#     }
#     pickle.dump(debug_data, f)

# Parameter containing:
# tensor([[ 4.3736e-05,  2.5323e-05, -4.5608e-05,  ...,  2.2342e-05,
#           1.6306e-04, -3.1887e-06],
#         [-1.2287e-04, -1.3227e-04,  2.0719e-05,  ...,  1.2726e-04,
#          -1.3207e-04, -1.2113e-04],
#         [ 3.8978e-06,  3.3790e-06, -2.6135e-06,  ..., -2.1652e-07,
#           2.6649e-06, -4.4819e-06],
#         ...,
#         [-4.1414e-06,  3.1061e-06,  3.1812e-07,  ...,  3.3305e-06,
#           2.4226e-06, -4.7595e-07],
#         [ 1.3350e-06,  2.6948e-06, -4.0382e-06,  ...,  3.1915e-06,
#          -8.1440e-07, -2.9296e-06],
#         [-3.9488e-06,  3.0671e-07,  2.6787e-06,  ..., -4.4720e-06,
#          -1.2872e-06, -4.5013e-06]], requires_grad=True)

# target_model
# Parameter containing:
# tensor([[-3.3223e-06,  1.8019e-06,  3.1530e-06,  ...,  4.3003e-06,
#          -3.4874e-06,  1.2513e-06],
#         [ 7.8425e-07, -4.0294e-06, -1.3495e-06,  ...,  3.4934e-06,
#          -4.3283e-06,  4.5704e-06],
#         [ 3.8978e-06,  3.3790e-06, -2.6135e-06,  ..., -2.1652e-07,
#           2.6649e-06, -4.4819e-06],
#         ...,
#         [-4.1414e-06,  3.1061e-06,  3.1812e-07,  ...,  3.3305e-06,
#           2.4226e-06, -4.7595e-07],
#         [ 1.3350e-06,  2.6948e-06, -4.0382e-06,  ...,  3.1915e-06,
#          -8.1440e-07, -2.9296e-06],
#         [-3.9488e-06,  3.0671e-07,  2.6787e-06,  ..., -4.4720e-06,
#          -1.2872e-06, -4.5013e-06]], requires_grad=True)