import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append('/hpcgpfs01/scratch/gzhao/rl-prompt')
from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model)
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)

from fsc_helpers import (PromptedClassificationRewardConfig,
                         FewShotClassificationDatasetConfig,
                         make_prompted_classification_reward,
                         make_few_shot_classification_dataset)


# Compose default config
config_list = [PromptedClassificationRewardConfig,
                FewShotClassificationDatasetConfig, LMAdaptorModelConfig,
                SinglePromptModelConfig, SQLModuleConfig, TrainerConfig]
cs = compose_hydra_config_store('base_fsc', config_list)


@hydra.main(version_base=None, config_path="./", config_name="fsc_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    (train_dataset, val_dataset, test_dataset,
     num_classes, verbalizers, template) = \
        make_few_shot_classification_dataset(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    # val_dataset = test_dataset
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    reward = make_prompted_classification_reward(num_classes, verbalizers, 
                                                 template, config)
    algo_module = make_sql_module(prompt_model, reward, config)   # sql model, prompt model and policy model are all the models generate prompts
    
    # task lm is the model to classify the review sentiment

    # Hack for few-shot classification - Each batch contains all examples
    config.train_batch_size = len(train_dataset)
    config.eval_batch_size = len(val_dataset)
    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    trainer.train(config=config, run_name=config.run_name)


if __name__ == "__main__":
    main()


"""
/hpcgpfs01/scratch/gzhao/rl-prompt/examples/few-shot-classification/fsc_reward.py 
calcualte the reward and print the results:
2 | BrowserStats 
 Class 0 Example: Horrible management. Doors were supposed to open at 12:30 - didn't open until 12:50. Snack bar not ready. Manager wasn't smart enough to apologize. Only open seating - seats very uncomfortable. BrowserStats <mask> . | Probs: [0.41, 0.46, 0.01, 0.06, 0.05] 
 Class 1 Example: The pizza was bad. The problem begins with ordering. The menu gives a few choice of sauce plus a lot of topping. It does not offer any guidance how to create a good pizza. A few signature pizza on the menu would be a welcomes improvement. \n\nThe price adds up quickly with multiple toppings.   I would recommend any fast food joints over this place. BrowserStats <mask> . | Probs: [0.02, 0.5, 0.02, 0.36, 0.1] 
 Class 2 Example: .59 fountain drinks! Can't go wrong! Weird people at night, but that's any gas station. BrowserStats <mask> . | Probs: [0.05, 0.16, 0.16, 0.33, 0.3] 
 Class 3 Example: Meh, its Walmart. My bf likes going here, he like has wet dreams over this store... I told him just to work here because we go like once a week! Its all the same to me, all I usually go to buy is TP because I like support the non jerks to their employee type of store.\n\nIts nice if you wanna just go to one place and like older produce and not fresh meat! BrowserStats <mask> . | Probs: [0.01, 0.27, 0.1, 0.43, 0.19] 
 Class 4 Example: I've been coming here for a while now and have to say im never let down. Food and service have always been impeccable. BrowserStats <mask> . | Probs: [0.02, 0.07, 0.03, 0.39, 0.49] 
 Accuracy: 0.45000001788139343 | Reward: -12.81

(round num | suggested prompt
five examples of reviews and probs for the review
Accuracy and reward of the review
)



"""