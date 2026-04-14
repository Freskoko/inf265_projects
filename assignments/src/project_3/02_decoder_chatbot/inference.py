import numpy as np
import torch

def greedy_sampling(last_token_logits):
    # greedy sampling (input is the logits of the last token, output is the selected token ID)
    return torch.argmax(last_token_logits)

def top_p_sampling(last_token_logits, p=0.95, temperature=0.7):
    # Implement top-p sampling with temperature (input is the logits of the last token, output is the selected token ID)

    # softmax
    logits = last_token_logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # get p
    # gives both sorted prob and indicies
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    where_threshold_reached = cumulative_probs <= p

    # we need to pick atleast something (what if we never reach threshold!)
    where_threshold_reached[0] = True

    filtered_probs = sorted_probs[where_threshold_reached]
    filtered_indices = sorted_indices[where_threshold_reached]

    filtered_probs = filtered_probs / filtered_probs.sum()

    # get logits back
    next_token = torch.multinomial(filtered_probs, 1)

    return filtered_indices[next_token]


def sample_sequence(input_sequence, model, strategy, max_len, device, end_id, p=0.95, temperature=0.7):
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(device) # Add batch dimension and move to device
        answer = []
        for _ in range(max_len):
            last_token_logits = model(input_sequence)
            last_token_logits = last_token_logits[0, -1, :]

            if strategy == "greedy":
                next_token = greedy_sampling(last_token_logits)
            elif strategy == "top-p":
                next_token = top_p_sampling(last_token_logits, p=p, temperature=temperature)
            else:
                raise ValueError("Invalid sampling strategy.")

            input_sequence = torch.cat([input_sequence, next_token.view(1, 1)], dim=1)
            answer.append(next_token.item())

            if next_token == end_id or input_sequence.size(1) >= max_len:
                break

        return answer

def tokenize_input(tokenizer, text, sep_id):
    """
    Tokenize input text and add special tokens.
    """
    tokens = tokenizer.encode(text).ids
    tokens = tokens + [sep_id]
    return torch.tensor(tokens)

def decode_output(tokenizer, tokens):
    """
    Decode output tokens.
    """
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    from config import config
    from tokenizers import Tokenizer
    from model import TransformerModel

    model = TransformerModel(config)
    model = model.to(config.device)
    model = torch.compile(model)
    model.load_state_dict(torch.load(config.model_filename, weights_only=True, map_location=config.device))

    tokenizer = Tokenizer.from_file(config.tokenizer_filename)

    sep_id = tokenizer.token_to_id(config.sep_token)
    end_id = tokenizer.token_to_id(config.end_token)

    question_text = "what is the largest dog breed?"

    input_sequence = tokenize_input(tokenizer, question_text, sep_id)

    print("Greedy sampling:")
    answer = sample_sequence(input_sequence, model, "greedy", 100, config.device, end_id)
    answer_text = decode_output(tokenizer, answer)
    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")

    print("Top-p sampling (p=0.95, temperature=0.7):")
    answer = sample_sequence(input_sequence, model, "top-p", 100, config.device, end_id, p=0.95, temperature=0.7)
    answer_text = decode_output(tokenizer, answer)
    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")


