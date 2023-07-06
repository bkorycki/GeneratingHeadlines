import torch
from torch.utils.data import Dataset
import pandas as pd
import re, html, unicodedata
from unidecode import unidecode
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List


class SimpleNewsDataset(Dataset):
	"""
	Dataset consisting of article descriptions
	"""

	def __init__(self, df, tokenizer):
		BOS = tokenizer.bos_token
		EOS = tokenizer.eos_token

		self.token_ids = []
		self.attn_masks = []
		self.df = df

		max_len = max(df.description_tokens.map(len))
		
		for _, desc in df['description'].items():
			text = BOS + desc + EOS
			encodings_dict = tokenizer(text, truncation=True, max_length=max_len, padding="max_length")

			self.token_ids.append(torch.tensor(encodings_dict['input_ids']))
			self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
			
	def __len__(self):
		return len(self.token_ids)

	def __getitem__(self, idx):
		return self.token_ids[idx], self.attn_masks[idx] 


class ContextualNewsDataset(Dataset):

	def __init__(self, df, tokenizer, target: str):
		""" 
		Dataset for contextual text generation. 
		-target: 
			'desc': For generating an article description given it's title
			'title': For generating a title given it's description
		"""
		BOS = tokenizer.bos_token
		SEP = tokenizer.sep_token
		PAD = tokenizer.pad_token

		target_tokens, context_tokens = "title_tokens", "description_tokens"
		if target == "desc":
			target_tokens, context_tokens = "description_tokens", "title_tokens"
		elif target != "title":
			raise ValueError("Target must be set to 'title' or 'desc'.")

		tokens = df.apply(lambda x:  x[context_tokens] + [SEP] + x[target_tokens] + [BOS], axis=1)
		max_len = max(tokens.map(len))
		tokens = tokens.apply(lambda x: x + [PAD]* (max_len-len(x)))
		
		self.df = df
		self.token_ids = list(tokens.map(tokenizer.convert_tokens_to_ids))
		self.sep_pos = list(df[context_tokens].map(len))

	def __len__(self):
		return len(self.token_ids)

	def __getitem__(self, idx):  
		token_ids = torch.tensor(self.token_ids[idx])
		sample = {'token_ids': token_ids, 'sep_pos': self.sep_pos[idx]}
		return sample


def get_datasets(config, tokenizer) -> List[Dataset]:
	""" Returns [train_dataset, val_dataset, test_dataset] based on split defined in config."""
	if config.get("TARGET_TYPE"):
		df = load_dataframe(tokenizer)
	else:
		df = load_dataframe(tokenizer, contextual=False)
	# Split data
	df_train, df_test = train_test_split(df, train_size=int(config["TRAIN_SPLIT"]*len(df)), random_state=config["RANDOM_SEED"])
	df_test, df_val = train_test_split(df_test, train_size=int(config["TEST_SPLIT"]*len(df)), random_state=config["RANDOM_SEED"])

	datasets = []
	for sub_df in [df_train, df_val, df_test]:
		if config.get("TARGET_TYPE"):
			dataset = ContextualNewsDataset(sub_df, tokenizer, config["TARGET_TYPE"])
		else:
			dataset = SimpleNewsDataset(sub_df, tokenizer)
		datasets.append(dataset)
	return datasets


def load_dataframe(tokenizer, contextual=True):
	""" Loads and cleans data.
		If contextual=false, only keep 'description'
	"""
	# Load data frame
	df = pd.read_json('data/data.jsonl')
	print(f"{len(df)} articles loaded.")	
	
	# Normalize
	df['title'] = df["title"].map(normalize_text)
	df['description'] = df["description"].map(normalize_text)

	# Filter out vietnamese and drop duplicates
	df = df[~df["url"].str.contains("thanhnien")][["title", "description"]]
	df.drop_duplicates(subset='title', inplace=True, keep='first')
	df.drop_duplicates(subset='description', inplace=True, keep='first')

	# Tokenize
	df['title_tokens'] = df["title"].map(tokenizer.tokenize)
	df['description_tokens'] = df["description"].map(tokenizer.tokenize)

	df = remove_outliers(df, contextual)

	print(f"{len(df)} samples after cleaning")
	return  df

def normalize_text(text, form='NFC') -> str:
	# Remove URLs
	text = re.sub(r'https?://\S+|www\.\S+', '', text)
	# Normalize 
	text = html.unescape(text)
	text = unicodedata.normalize(form, text) 
	text = unidecode(text)
	# White space
	text = re.sub(r'\s+', ' ', text)

	return text

def remove_outliers(df, contextual):
	# Remove rows with empty title/desc 
	df = df[df.description_tokens.map(len) > 0]
	cols = ["description_tokens"]
	if contextual:
		df = df[df.title_tokens.map(len) > 0]
		# Only keep rows with title shorter than desc.
		df = df[df.title_tokens.map(len) < df.description_tokens.map(len)]
		cols = ["title_tokens","description_tokens"]
	# Remove outliers
	for col in cols:
		series = df[col].map(len)
		mu, std = np.mean(series), np.std(series)
		df = df[series.between(mu - 3*std, mu + 3*std)]
	df.reset_index(inplace=True, drop=True)
	return df
