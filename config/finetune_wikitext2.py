import time

eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'wikitext-2'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'wikitext-2'
init_from = 'gpt2' # this is the largest GPT-2 model
out_dir = 'ckpt/out-wiki/'+init_from


# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# wikitext-2 has 2M(2,459,198) tokens, so 1 epoch ~= 75 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 150

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
