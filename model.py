import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModel


class Parser(nn.Module):

    def __init__(
        self,
        tokenizer,
        embedding_layer,
        decimal_symbol=".",
        hidden_size=300,
        num_layers=2,
        num_classes=3
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer   # nn.Embedding aus dem Sprachmodell
        self.decimal_symbol = decimal_symbol

        ds = re.escape(decimal_symbol)
        self._split_pattern = rf'(?<![\d {ds}])(?=\d)|(?<=\d)(?![\d {ds}])'

        input_size = embedding_layer.embedding_dim

        # ----------------------------------------------------
        # Bidirectional LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
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

            nn.Linear(128, num_classes),

            nn.Softmax(dim=-1)
        )

    def embed(self, text):
        """Regex-Split + Embedding-Lookup. Gibt (embeddings, number_mask, values) zurück."""
        embeddings = []
        number_mask = []
        values = []

        for seg in re.split(self._split_pattern, text):
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
        values = np.array(values)
        return embeddings, number_mask, values

    def forward(self, text):
        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        embeddings, number_mask, values = self.embed(text)
        input_tensor = embeddings.unsqueeze(1)   # (seq_len, 1, embedding_size)

        # ----------------------------------------------------
        # LSTM + Classifier
        # ----------------------------------------------------

        output, (hn, cn) = self.lstm(input_tensor)
        predictions = self.classifier(output).squeeze(1)   # (seq_len, num_classes)

        # ----------------------------------------------------
        # Nur Zahlen-Positionen zurückgeben
        # ----------------------------------------------------

        number_predictions = predictions[number_mask]

        return number_predictions, number_mask, values

model_name = 'intfloat/multilingual-e5-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)

model = Parser(
    tokenizer,
    embedding_model.get_input_embeddings()
    )