import os
import yaml 
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def get_tokenizer(size: str='small', contextual=True):
	if contextual:
		special_tokens_dict = {'sep_token': '<|sep|>', 'pad_token': '<|pad|>', 'bos_token': '<|endoftext|>'}
	else:
		special_tokens_dict = {'pad_token': '<|pad|>', 'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>'}

	gpt_type = get_pretrained_name(size)	
	tokenizer = GPT2Tokenizer.from_pretrained(gpt_type)
	tokenizer.add_special_tokens(special_tokens_dict)
	return tokenizer


def init_model(tokenizer, size: str='small'):
	gpt_name = get_pretrained_name(size)	
	model = GPT2LMHeadModel.from_pretrained(gpt_name)
	model.resize_token_embeddings(len(tokenizer))

	return model

def load_model(tokenizer, config, epoch: int):
	model_path = f"./output/{config['exp_name']}/models/checkpoint_epoch{epoch}.pt"
	checkpoint = torch.load(model_path)

	model = init_model(tokenizer, config['GPT_SIZE'])
	model.load_state_dict(checkpoint['model_state_dict'])
	# model.load_state_dict(torch.load(f'./output/models/{model_name}.pt'))
	model.eval()
	return model


def get_pretrained_name(size: str) -> str:
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

def load_config(experiment_name):
	with open(f"./config/{experiment_name}.yaml", 'r') as file:
		config = yaml.safe_load(file)
	config["exp_name"] = experiment_name
	return config