import re

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


class Parser(nn.Module):

    def __init__(
        self,
        embeddingModel="sentence-transformers/all-MiniLM-L6-v2",
        decimal_symbol=".",
        hidden_size=300,
        num_layers=2,
        num_classes=4
    ):
        super().__init__()

        self.decimal_symbol = decimal_symbol
        self.tokenizer = AutoTokenizer.from_pretrained(embeddingModel)
        print("loaded tokenizer")
        self.embedding_layer = AutoModel.from_pretrained(embeddingModel).get_input_embeddings()
        print("loaded embedding")

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

    # ----------------------------------------------------
    # EXPORTIERBARER KERN — nur Tensor-Operationen, kein Text/Tokenizer
    # ----------------------------------------------------

    def forward(self, token_ids: torch.LongTensor, number_mask: torch.BoolTensor):
        # token_ids:   (seq_len,) — an Zahlen-Positionen beliebige Dummy-ID (z.B. 0)
        # number_mask: (seq_len,) — True an Zahlen-Positionen

        embeddings = self.embedding_layer(token_ids)                  # (seq_len, embedding_dim)

        zeros = torch.zeros_like(embeddings)
        embeddings = torch.where(number_mask.unsqueeze(-1), zeros, embeddings)

        input_tensor = embeddings.unsqueeze(1)                        # (seq_len, 1, embedding_size)

        output, (hn, cn) = self.lstm(input_tensor)
        predictions = self.classifier(output).squeeze(1)              # (seq_len, num_classes)

        return predictions                     

    # ----------------------------------------------------
    # PREPROCESSING — bleibt in Python, wird NICHT exportiert
    # Nutzt du zum Trainieren, Testen, oder um token_ids/number_mask
    # für forward() bzw. den Dummy-Input beim Export zu erzeugen
    # ----------------------------------------------------

    def prepare_input(self, text):
        """Regex-Split + Tokenisierung. Gibt (token_ids, number_mask, values) zurück."""

        ds = re.escape(self.decimal_symbol)
        token_ids = []
        number_mask = []
        values = []

        for seg in re.split(rf'(?<![\d{ds}])(?=\d)|(?<=\d)(?![\d{ds}])', text):
            if seg != "" and seg[0].isdigit():
                token_ids.append(0)  # Dummy-ID, wird in forward() weggenullt
                number_mask.append(True)
                values.append(seg)
            elif seg != "":
                ids = self.tokenizer.encode(seg, add_special_tokens=False)
                for tid in ids:
                    token_ids.append(tid)
                    number_mask.append(False)
                    values.append(self.tokenizer.decode(tid))

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        number_mask = torch.tensor(number_mask, dtype=torch.bool)

        return token_ids, number_mask, values
    
    def evalText(self, text):
        tokens, mask, vals = self.prepare_input(text)

        prediction = self.__call__(tokens, mask)

        return prediction[mask], vals