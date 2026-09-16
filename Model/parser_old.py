import re

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel

class Parser(nn.Module):

    def __init__(
        self,
        embeddingModel = 'intfloat/multilingual-e5-base', 
        decimal_symbol = ".",
        hidden_size = 300,
        num_layers = 2,
        num_classes = 4
    ):
        super().__init__()

        self.decimal_symbol = decimal_symbol
            
        self.tokenizer = AutoTokenizer.from_pretrained(embeddingModel)
        self.embedding_layer = AutoModel.from_pretrained(embeddingModel).get_input_embeddings()

        self.embedding_layer.requires_grad_(False)  # vortrainiertes Embedding einfrieren

        # ----------------------------------------------------
        # Bidirectional LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=self.embedding_layer.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True
        )

        lstm_output_size = hidden_size * 2

        # ----------------------------------------------------
        # Neural Network auf jedem LSTM-State
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def dim(self):
            return self.embedding_layer.embedding_dim

    def forward(self, text):

        embedding, number_mask, _ = self.embed(text)

        input_tensor = embedding.unsqueeze(1)   # (seq_len, 1, embedding_size)

        output, (hn, cn) = self.lstm(input_tensor)
        predictions = self.classifier(output).squeeze(1)   # (seq_len, num_classes)

        return predictions[number_mask]

    def embed(self, text):    
        """Regex-Split + Embedding-Lookup. Gibt (embeddings, number_mask, values) zurück."""

        ds = re.escape(self.decimal_symbol)
        
        embeddings = []
        number_mask = []
        values = []

        for seg in re.split(rf'(?<![\d{ds}])(?=\d)|(?<=\d)(?![\d{ds}])', text):
            if seg != "" and seg[0].isdigit():
                embeddings.append(torch.zeros(self.embedding_layer.embedding_dim))
                number_mask.append(True)
                values.append(seg)
            else:
                token_ids = self.tokenizer.encode(seg, add_special_tokens=False)
                for tid in token_ids:
                    embeddings.append(self.embedding_layer(torch.tensor([tid])).squeeze(0))
                    number_mask.append(False)
                    values.append(self.tokenizer.decode(tid))

        embeddings = torch.stack(embeddings)
        number_mask = torch.tensor(number_mask, dtype=torch.bool)

        return embeddings, number_mask, values

