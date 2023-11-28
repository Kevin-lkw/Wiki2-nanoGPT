# wiki-2 dataset
dataset = 'wikitext-2'
split = 'test'
# test origin model
init_from = 'gpt2-medium'


# test finetuned model
# init_from = 'resume'
# out_dir = 'ckpt/out-wiki/gpt2'

# considering efficiency and accuracy, we use stride = 128
# feel free to change it 
stride=1024

