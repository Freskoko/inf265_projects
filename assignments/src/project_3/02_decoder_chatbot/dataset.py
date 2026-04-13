import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

class QADataset(Dataset):
    def __init__(
        self, config, tokenizer,
    ):
        self.dataset = load_from_disk(config.dataset)
        n_subset = int(config.model_train_fraction * len(self.dataset))
        self.dataset= self.dataset.select(range(n_subset))
        print(
            f"Loaded dataset of size {len(self.dataset)} with columns {self.dataset.column_names}"
        )

        self.tokenizer = tokenizer
        self.max_length = config.max_len

        # Special token IDs (you can use these IDs in the __getitem__ method)
        self.pad_id = self.tokenizer.token_to_id(config.pad_token)
        self.sep_id = self.tokenizer.token_to_id(config.sep_token)
        self.end_id = self.tokenizer.token_to_id(config.end_token)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question, answer = self.dataset[idx]["question"], self.dataset[idx]["answer"]

        # question
        tokenized_question = self.tokenizer.encode(question)
        ids_question = tokenized_question.ids

        # answer:
        tokenized_answer = self.tokenizer.encode(answer)
        ids_answer = tokenized_answer.ids

        # combined
        ids_src = ids_question + [self.pad_id] + ids_answer

        # end token
        ids_src += self.end_id

        # padding / truncate
        ids_src = ids_src[:self.max_length]
        length_needed_padding = self.max_length - len(ids_src)
        ids_src += [self.pad_idx] * length_needed_padding

        # target is source 1 pos forward
        source = torch.tensor(ids_src, dtype=torch.long)
        target = source.clone()
        target[:-1] = source[1:]
        target[-1] = self.pad_id

        # padding mask
        padding_mask = (source == self.pad_id)
        # set to -100 in target for padding, not important
        target[padding_mask] = -100

        return {
            "source_sequence": source,
            "target_sequence": target,
            "key_padding_mask": padding_mask,
        }


if __name__ == "__main__":
    from config import config
    from tokenizers import Tokenizer

    # Sanity check the dataset class
    print(config.tokenizer_filename)
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    idx = 1
    config.max_len = 64 # For testing purposes
    dataset = QADataset(config, tokenizer)

    source, target, key_padding_mask = dataset[idx].values()

    print("Source sequence shape:", source.shape)
    print("Target sequence shape:", target.shape)
    print("Key padding mask shape:", key_padding_mask.shape)

    print("Source sequence:", source)
    print("Target sequence:", target)
    print("Key padding mask:", key_padding_mask)

    decoded_source = tokenizer.decode(source.tolist(), skip_special_tokens=False)
    decoded_target = tokenizer.decode(target[target != -100].tolist(), skip_special_tokens=False)
    print("Decoded source sequence:", decoded_source)
    print("Decoded target sequence:", decoded_target)

