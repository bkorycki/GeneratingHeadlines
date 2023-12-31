import torch
import torch.nn.functional as F


def generate_sample(data, tokenizer, model, num=1, length=20, temperature=1, top_k=0, top_p=0.5, device=torch.device('cuda')):
    # Set seed in between subsequent calls for reproducibility
    torch.manual_seed(42) 
    torch.use_deterministic_algorithms(False)

    for i in range(num):
        print("*"*50)
        sample = data[i+3]
        idx = sample['sep_pos']

        context = sample['token_ids'][:idx].tolist()
        target = sample['token_ids'][idx+1:][:100].tolist()

        generated_tokens = sample_seq(model, context, length, device, temperature, top_k, top_p)
        generated_tokens = generated_tokens[0, idx:].tolist()
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print('CONTEXT:\n', tokenizer.decode(context, skip_special_tokens=True))
        print("\nGENERATED TEXT:\n", generated_text)
        print('TRUE TARGET:\n', tokenizer.decode(target,skip_special_tokens=True),"   \n")


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_seq(model, context, length, device, temperature=1, top_k=0, top_p=0.0):
    """ Generates a sequence of tokens 
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():  
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def generate_beam_sample(data, tokenizer, model, num=1, length=20, beam_size=3, device=torch.device('cuda')):
    for i in range(num):
        sample = data[i]
        idx = sample['sep_pos']
        context = sample['token_ids'][:idx].tolist()
        summary = sample['token_ids'][idx+1:][:100].tolist()
        scores, sequences = beam_search(model, context, length, beam_size, device)
        print('description', end='\n\n')
        print(tokenizer.decode(context[:-1]), end='\n\n')
        # print('actual_summary', end='\n\n')
        # print(tokenizer.decode(summary), end='\n\n')
        for i in range(len(sequences)):
            text = tokenizer.convert_ids_to_tokens(sequences[i],skip_special_tokens=True)
            text = tokenizer.convert_tokens_to_string(text)  
            print("generated_summary-{} and Score is {}.".format(i+1, scores[i]), end='\n\n')
            print(text, end='\n\n')

def beam_search(model, context, length, beam_size, device, temperature=1):
    """ Generate sequence using beam search https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            beam_size: >=1 and <= total_no_of_tokens
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
    """
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    with torch.no_grad():  
        inputs = {'input_ids': context}
        outputs = model(**inputs) 
        next_token_logits = outputs[0][0, -1, :] / temperature
        next_token_probs = F.softmax(next_token_logits)
        scores, indices = torch.topk(next_token_probs, beam_size)
        indices = indices.tolist()
        sequences = [[c] for c in indices]
        for _ in range(length-1):
            logits = torch.zeros(beam_size*len(next_token_logits))
            for j in range(len(sequences)):
                new_generated = torch.cat((context,torch.tensor([sequences[j]], dtype=torch.long, device=device)),dim=1)
                inputs = {'input_ids': new_generated}
                outputs = model(**inputs) 
                next_token_logits = outputs[0][0, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits)
                start, stop = j*len(next_token_logits), (j+1)*len(next_token_logits)
                logits[start:stop] = scores[j]*next_token_probs
            scores, new_logits_indices = torch.topk(logits,beam_size)
            logits = (new_logits_indices%50259).tolist()
            for j in range(len(sequences)):
                sequences[j] = sequences[j]+[logits[j]]
    return scores, sequences