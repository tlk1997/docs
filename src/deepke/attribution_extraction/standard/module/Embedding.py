import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, config):
        """
        Args:
            word embedding: Generally 0 is padding
            pos embedding:  Generally 0 is padding
            dim_strategy: [cat, sum], Whether multiple embeddings are spliced or added
        """
        super(Embedding, self).__init__()

        # self.xxx = config.xxx
        self.vocab_size = config.vocab_size
        self.word_dim = config.word_dim
        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim if config.dim_strategy == 'cat' else config.word_dim
        self.dim_strategy = config.dim_strategy

        self.wordEmbed = nn.Embedding(self.vocab_size, self.word_dim, padding_idx=0)
        self.entityPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)
        self.attribute_keyPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)
        
        self.layer_norm = nn.LayerNorm(self.word_dim)

    def forward(self, *x):
        word, entity, attribute_key = x
        word_embedding = self.wordEmbed(word)
        entity_embedding = self.entityPosEmbed(entity)
        attribute_key_embedding = self.attribute_keyPosEmbed(attribute_key)

        if self.dim_strategy == 'cat':
            return torch.cat((word_embedding, entity_embedding, attribute_key_embedding), -1)
        elif self.dim_strategy == 'sum':
            # 此时 pos_dim == word_dim
            return self.layer_norm(word_embedding + entity_embedding + attribute_key_embedding)
        else:
            raise Exception('dim_strategy must choose from [sum, cat]')
