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
        