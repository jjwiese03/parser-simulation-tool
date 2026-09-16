"""
generate_scheiben_data.py

Generiert synthetische Trainingsdaten für ein LSTM, das Zahlenwerte in
strukturiertem Text (JSON, Python-Dict/Liste, CSV, key=value) klassifizieren
soll.

Jede Ausgabezeile ist ein Python-Tupel: (text, labels)
  - text:   String, der ein "Ensemble" von n Scheiben in einem von mehreren
            Formaten beschreibt.
  - labels: Liste von ints, ein Label pro Zahl im Text, in der Reihenfolge
            ihres Auftretens (links nach rechts).

Label-Bedeutung:
  1 = Position   (absolute Lage der Scheibe)
  2 = Distanz    (Abstand zur vorherigen Scheibe)
  3 = Dicke      (Materialstärke der Scheibe)
  0 = Sonstiges  (Epsilon, Zählindex, "n"-Feld - keine der drei Zielgrößen)

Nutzung:
  python generate_scheiben_data.py
  python generate_scheiben_data.py --n_samples 20000 --seed 123 --output daten.txt
"""

import argparse
import random
import re
import ast


# ---------------------------------------------------------------------------
# Hilfsklasse: baut den Text und die dazugehörige Label-Liste synchron auf
# ---------------------------------------------------------------------------
class Builder:
    def __init__(self):
        self.parts = []
        self.labels = []

    def text(self, s):
        """Fügt reinen Text hinzu (Klammern, Kommas, Keys, ...) - kein Label."""
        self.parts.append(s)

    def number(self, value_str, label):
        """Fügt eine Zahl als Text hinzu und hängt das zugehörige Label an."""
        self.parts.append(value_str)
        self.labels.append(label)

    def build(self):
        return "".join(self.parts), self.labels


def fmt(v):
    """Formatiert eine Zahl mit zufälliger Anzahl Nachkommastellen."""
    decimals = random.choice([0, 1, 1, 2, 2, 3])
    if decimals == 0:
        return str(int(round(v)))
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Erzeugung der physikalischen Rohdaten für ein Scheiben-Ensemble
# ---------------------------------------------------------------------------
def gen_ensemble(n_min=3, n_max=12):
    n = random.randint(n_min, n_max)
    mode = random.choices(["absolute", "relative", "both"], weights=[0.45, 0.45, 0.10])[0]

    positions = None
    distances = None

    if mode in ("absolute", "both"):
        cum = 0.0
        positions = []
        for _ in range(n):
            cum += random.uniform(3, 60)
            positions.append(round(cum, 3))

    if mode in ("relative", "both"):
        distances = [round(random.uniform(0.5, 50), 3) for _ in range(n - 1)]

    widths = [round(random.uniform(0.05, 10), 3) for _ in range(n)]

    epsilon = None
    if random.random() < 0.25:
        epsilon = [round(random.uniform(1.5, 12.0), 2) for _ in range(n)]

    include_meta = random.random() < 0.3  # "n"-Feld bzw. Zählindex mit ausgeben?

    return n, positions, distances, widths, epsilon, include_meta


# ---------------------------------------------------------------------------
# Renderer: jeder erzeugt denselben Datensatz in einem anderen Textformat
# ---------------------------------------------------------------------------
LABEL_MAP = {"positions": 1, "distances": 2, "widths": 3, "epsilon": 0}


def render_json(n, positions, distances, widths, epsilon, include_n):
    b = Builder()
    b.text("{")
    fields = []
    if include_n:
        fields.append(("n", None))
    if positions is not None:
        fields.append(("positions", positions))
    if distances is not None:
        fields.append(("distances", distances))
    fields.append(("widths", widths))
    if epsilon is not None:
        fields.append(("epsilon", epsilon))
    random.shuffle(fields)

    for i, (key, arr) in enumerate(fields):
        b.text(f'"{key}": ')
        if key == "n":
            b.number(str(n), 0)
        else:
            b.text("[")
            for j, v in enumerate(arr):
                b.number(fmt(v), LABEL_MAP[key])
                if j < len(arr) - 1:
                    b.text(", ")
            b.text("]")
        if i < len(fields) - 1:
            b.text(", ")
    b.text("}")
    return b.build()


def render_py_dict(n, positions, distances, widths, epsilon, include_n):
    b = Builder()
    b.text("{")
    fields = []
    if include_n:
        fields.append(("n", None))
    if positions is not None:
        fields.append(("positions", positions))
    if distances is not None:
        fields.append(("distances", distances))
    fields.append(("widths", widths))
    if epsilon is not None:
        fields.append(("epsilon", epsilon))
    random.shuffle(fields)

    for i, (key, arr) in enumerate(fields):
        b.text(f"'{key}': ")
        if key == "n":
            b.number(str(n), 0)
        else:
            b.text("[")
            for j, v in enumerate(arr):
                b.number(fmt(v), LABEL_MAP[key])
                if j < len(arr) - 1:
                    b.text(", ")
            b.text("]")
        if i < len(fields) - 1:
            b.text(", ")
    b.text("}")
    return b.build()


def render_discs_list(n, positions, distances, widths, epsilon, include_n):
    b = Builder()
    if include_n:
        b.text("# n = ")
        b.number(str(n), 0)
        b.text("\n")
    b.text("[")
    for i in range(n):
        b.text("{")
        keys = []
        if positions is not None:
            keys.append(("position", positions[i], 1))
        if distances is not None and i > 0:
            keys.append(("distance", distances[i - 1], 2))
        keys.append(("width", widths[i], 3))
        if epsilon is not None:
            keys.append(("epsilon", epsilon[i], 0))
        for j, (k, v, lab) in enumerate(keys):
            b.text(f"'{k}': ")
            b.number(fmt(v), lab)
            if j < len(keys) - 1:
                b.text(", ")
        b.text("}")
        if i < n - 1:
            b.text(", ")
    b.text("]")
    return b.build()


def render_csv(n, positions, distances, widths, epsilon, include_index):
    b = Builder()
    headers = []
    if include_index:
        headers.append("index")
    if positions is not None:
        headers.append("position")
    if distances is not None:
        headers.append("distance")
    headers.append("width")
    if epsilon is not None:
        headers.append("epsilon")
    b.text(",".join(headers) + "\n")

    for i in range(n):
        row = []
        if include_index:
            row.append(("num", i + 1, 0))
        if positions is not None:
            row.append(("num", positions[i], 1))
        if distances is not None:
            if i == 0:
                row.append(("empty", None, None))
            else:
                row.append(("num", distances[i - 1], 2))
        row.append(("num", widths[i], 3))
        if epsilon is not None:
            row.append(("num", epsilon[i], 0))

        for j, (kind, v, lab) in enumerate(row):
            if kind == "num":
                b.number(fmt(v), lab)
            if j < len(row) - 1:
                b.text(",")
        b.text("\n")
    return b.build()


def render_kv(n, positions, distances, widths, epsilon, include_n):
    b = Builder()
    if include_n:
        b.text("n = ")
        b.number(str(n), 0)
        b.text("\n")
    if positions is not None:
        b.text("positions = [")
        for j, v in enumerate(positions):
            b.number(fmt(v), 1)
            if j < len(positions) - 1:
                b.text(", ")
        b.text("]\n")
    if distances is not None:
        b.text("distances = [")
        for j, v in enumerate(distances):
            b.number(fmt(v), 2)
            if j < len(distances) - 1:
                b.text(", ")
        b.text("]\n")
    b.text("widths = [")
    for j, v in enumerate(widths):
        b.number(fmt(v), 3)
        if j < len(widths) - 1:
            b.text(", ")
    b.text("]")
    if epsilon is not None:
        b.text("\nepsilon = [")
        for j, v in enumerate(epsilon):
            b.number(fmt(v), 0)
            if j < len(epsilon) - 1:
                b.text(", ")
        b.text("]")
    return b.build()


RENDERERS = [render_json, render_py_dict, render_discs_list, render_csv, render_kv]
WEIGHTS = [0.25, 0.20, 0.20, 0.20, 0.15]


def make_sample(n_min, n_max):
    n, positions, distances, widths, epsilon, include_meta = gen_ensemble(n_min, n_max)
    renderer = random.choices(RENDERERS, weights=WEIGHTS)[0]
    text, labels = renderer(n, positions, distances, widths, epsilon, include_meta)
    return text, labels


# ---------------------------------------------------------------------------
# Validierung: prüft, dass Zahlenanzahl im Text == Länge der Label-Liste
# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"-?\d+\.?\d*")


def validate(path):
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_numbers = 0
    mismatches = 0
    n_lines = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            text, labels = ast.literal_eval(line.strip())
            found = NUM_RE.findall(text)
            if len(found) != len(labels):
                mismatches += 1
            total_numbers += len(labels)
            for l in labels:
                class_counts[l] += 1

    print(f"Zeilen geprüft:              {n_lines}")
    print(f"Mismatches (Zahl vs Label):  {mismatches}")
    print(f"Gesamtzahl gelabelter Zahlen:{total_numbers}")
    print(f"Klassenverteilung:           {class_counts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generiert Scheiben-Ensemble-Trainingsdaten für ein LSTM.")
    parser.add_argument("--n_samples", type=int, default=5000, help="Anzahl zu generierender Datenzeilen")
    parser.add_argument("--n_min", type=int, default=3, help="Minimale Anzahl Scheiben pro Ensemble")
    parser.add_argument("--n_max", type=int, default=12, help="Maximale Anzahl Scheiben pro Ensemble")
    parser.add_argument("--seed", type=int, default=42, help="Zufalls-Seed für Reproduzierbarkeit")
    parser.add_argument("--output", type=str, default="../genDataComplicated.txt", help="Pfad der Ausgabedatei")
    parser.add_argument("--no_validate", action="store_true", help="Validierung nach dem Schreiben überspringen")
    args = parser.parse_args()

    random.seed(args.seed)

    samples = [make_sample(args.n_min, args.n_max) for _ in range(args.n_samples)]

    with open(args.output, "w", encoding="utf-8") as f:
        for text, labels in samples:
            f.write(f"({text!r}, {labels})\n")

    print(f"Geschrieben: {len(samples)} Zeilen -> {args.output}")

    if not args.no_validate:
        validate(args.output)


if __name__ == "__main__":
    main()