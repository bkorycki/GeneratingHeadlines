# Directory structure

**Finetuning scripts**
- `nocontext_finetune_gpt.ipynb` finetune gpt2 on article descriptions
- `finetune_gpt.ipynb`	finetune gpt2 for conditional text generation 
	- `target = 'description'` given a title, generate the description
	- `target = 'title'` given the description, generate a headline

**Generation script**
- `generate.ipynb` 

**Output** 
- `output/logs/` TensorBoard training logs from different experiments