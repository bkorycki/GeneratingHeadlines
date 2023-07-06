# Generating Article Headlines with Limited Data using GPT-2
This repository contains the data and code for the following tasks:
1. Fine-tuning GPT2 on article summaries:  `nocontext_finetune_gpt.ipynb` 
2. In-Context Learning: `finetune_gpt.ipynb`
	a) Fine-tune GPT2 to generate article summary given it's title
	 `target='desc'` in `config/CONFIG.yaml` file
	b) Fine-tune GPT2 to predict a headline given an article summary
	 `target='title'` in config file

## In-Context Learning
- I follow a **text-summarization** approach outlined in *Sample Efficient Text Summarization Using a Single Pre-Trained Transformer* (Urvashi et al.)
- Input format: `[source_text <SEP> target_text <BOS>]`*
	*Original GPT2 does not have a separate `EOS` token
- **Loss**: cross-entropy over just the target (in line with seq2seq models)
- Since GPT2 is so big, I mitigate memory restrictions via **gradient accumulation**

<!-- ## Directory structure -->
<!-- 
**Finetuning scripts**
- `nocontext_finetune_gpt.ipynb` finetune gpt2 on article descriptions
- `finetune_gpt.ipynb`	finetune gpt2 for conditional text generation 
	- `target = 'description'` given a title, generate the description
	- `target = 'title'` given the description, generate a headline

**Generation script**: `generate.ipynb` 

**Training logs**: `output/logs/` 
- TensorBoard training logs from different experiments
 -->

## Versions
- `python3.8.10`
- `torch.2.0.1`
- CUDA 11.7
