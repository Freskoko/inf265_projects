import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4*dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4*dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_size)

        self.mh1 = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.ln2 = nn.LayerNorm(embed_size)

        # build in dropout here
        self.mlpblock = MLPBlock(embed_size, dropout=dropout)


    def forward(self, x, attn_mask, padding_mask):
        # ---- l1
        x_l1 = self.ln1(x)
        x_l1 = self.mh1(
            query = x_l1,
            key = x_l1,
            value = x_l1,
            key_padding_mask = padding_mask,
            attn_mask = attn_mask,
            need_weights = False,
            is_causal = True
        )[0] # todo unsure about this, returns tuple
        # attn_output.transpose(1, 0), attn_output_weights

        x_l1 += x
        # ---- l2
        x_l2_norm = self.ln2(x_l1)
        x_l2 = self.mlpblock(x_l2_norm)
        x_l2 += x_l2_norm
        return x_l2


# TODO:  but make sure to move the positional encoding values to the 8 same device as the input sequence in the forward method.
class PositionalEncoding(nn.Module):
    """
    Positional encoding module: adds positional information to the input embeddings.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()
        super().__init__()
        # from attention is all you need part 3.5
        positions = torch.arange(0, max_len).unsqueeze(1)

        iarange = torch.arange(0, embed_size, 2)

        div_term =  (10000.0 **((iarange) / embed_size)) # is this 2 correct?

        pe = torch.zeros(max_len, embed_size)

        # sin to even
        pe[:, 0::2] = torch.sin(positions / div_term)

        # cos to odd
        pe[:, 1::2] = torch.cos(positions / div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pos_encodings",pe)

    def forward(self, x):
        # add positional
        #  to the input embeddings x
        x_end = x.size(1)
        x = x + self.pos_encodings[:, :x_end, :]
        # we want it all from pos 1 to end of x
        return x


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.embed_size
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.dropout_p = config.dropout_p
        self.num_heads = config.num_heads
        self.device = config.device

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.max_len)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.num_heads, self.dropout_p) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)

        # Precompute the causal mask and positional encoding
        self.register_buffer("causal_mask", self.generate_causal_mask(self.max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)

        return self.fc_out(x)

    def generate_causal_mask(self, seq_len):
        """
        Generates an upper triangular mask to prevent attending to future tokens.
        """
        matrix = torch.ones(seq_len, seq_len)
        matrix = torch.triu(matrix, diagonal=1)
        return matrix

if __name__ == "__main__":
    from tokenizers import Tokenizer
    from torch.nn.functional import cross_entropy

    from config import config
    from utils import get_num_params
    from dataset import QADataset

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    # Simple forward pass for sanity checking
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    source = dataset[0]["source_sequence"].unsqueeze(0)
    target = dataset[0]["target_sequence"].unsqueeze(0)
    padding_mask = dataset[0]["key_padding_mask"].unsqueeze(0)

    # Forward pass
    out = model(source, padding_mask)
    print("Output shape:", out.shape)
    print("Target shape:", target.shape)
    print("Loss mask shape:", padding_mask.shape)

    # Calculate loss
    loss = cross_entropy(out.transpose(1, 2), target)
    print("Loss:", loss.item())

