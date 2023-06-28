import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def get_tokenizer(size):
	BOS = '<|endoftext|>'
	PAD = '<|pad|>'
	SEP = '<|sep|>'

	special_tokens_dict = {'sep_token': SEP, 'pad_token': PAD, 'bos_token': BOS}

	gpt_type = get_pretrained_name(size)	
	tokenizer = GPT2Tokenizer.from_pretrained(gpt_type)
	num_add_toks = tokenizer.add_special_tokens(special_tokens_dict)
	return tokenizer


def init_model(tokenizer, size):
	gpt_name = get_pretrained_name(size)	
	model = GPT2LMHeadModel.from_pretrained(gpt_name)
	model.resize_token_embeddings(len(tokenizer))

	return model

def load_model(model_path, tokenizer, size):
	model = init_model(tokenizer, size)

	checkpoint = torch.load(model_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	# model.load_state_dict(torch.load(f'./output/models/{model_name}.pt'))
	model.eval()

	return model

def get_pretrained_name(size):
	if size == "small":
		gpt_type = 'gpt2'
	elif size == "medium":
		gpt_type = 'gpt2-medium'
	else:
		raise ValueError("GPT2 size must be small or medium")
	return gpt_type

def setup_exp_folders(experiment_name):
	experiment_dir = os.path.join("./output", experiment_name)
	os.mkdir(experiment_dir)
	log_dir = os.path.join(experiment_dir, "logs")
	model_dir = f"{experiment_dir}/models"
	os.mkdir(model_dir)
	return log_dir, model_dir

