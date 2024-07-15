from tst_modules import PromptedGenerator
from transformers import pipeline, AutoTokenizer
task_lm = 'distilgpt2'
template = '{prompt} "{sentence_1}" "'
end_punct = '"'
pad_token = '<|endoftext|>'
generator_device = 'cuda'
lower_outputs = True
control_output_length = True
tokenizer = AutoTokenizer.from_pretrained(task_lm)
print('hre3')
generator = PromptedGenerator(task_lm, template, end_punct,
                                           pad_token, generator_device,
                                           lower_outputs, control_output_length)