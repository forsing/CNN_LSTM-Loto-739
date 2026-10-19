#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Loto 7/39 — CNN-LSTM, V1.

Osnovna arhitektura:
    Conv1D(64, 3) -> MaxPool(2)
    -> Conv1D(32, 3) -> MaxPool(2)
    -> LSTM(64) -> LSTM(32)
    -> regresija raspodela.

Cela mreža trenira se zajedno.

Tri CSV-a obrađuju se nezavisno.
Završno treniranje koristi sve istorijske prelaze
za koje postoji prethodni prozor od 30 izvlačenja.

Bez slučajne inicijalizacije, mešanja podataka,
dropout-a, šuma i generisanja kombinacija.

Bez klasifikacione funkcije greške, zaokruživanja
i naknadnog pomeranja brojeva.
"""


import argparse
import csv
import hashlib
import math
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "Nedostaje biblioteka. Instalacija:\n"
        "python3 -m pip install numpy torch\n"
        f"Detalj: {exc}"
    )


DEFAULT_FILES = (
    "/data/loto7_4704_k83.csv",
    "/data/loto7_4704_k83_loto_2974.csv",
    "/data/loto7_4704_k83_loto_plus_1730.csv",
)

BALLS = 39
PICKS = 7
WINDOW = 30

VALIDATION_SIZE = 48
TEST_SIZE = 64
MIN_TRAIN_SAMPLES = 128

MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")


def configure_torch():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def read_draws(filename):
    path = Path(filename).expanduser()

    if not path.is_file():
        raise ValueError(f"Fajl ne postoji: {path}")

    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()

            if not line:
                continue

            if "," in line or ";" in line:
                delimiter = "," if "," in line else ";"
                fields = next(
                    csv.reader([line], delimiter=delimiter, strict=True)
                )
                fields = [field.strip() for field in fields]
            else:
                fields = re.split(r"\s+", line)

            if len(fields) != PICKS:
                raise ValueError(
                    f"Red {line_number}: očekivano 7 brojeva, "
                    f"pronađeno {len(fields)}."
                )

            if any(
                re.fullmatch(r"[0-9]+", field) is None
                for field in fields
            ):
                raise ValueError(
                    f"Red {line_number}: očekivani su celi brojevi "
                    "bez zaglavlja i dodatnih kolona."
                )

            numbers = [int(field) for field in fields]

            if any(number < 1 or number > BALLS for number in numbers):
                raise ValueError(
                    f"Red {line_number}: brojevi moraju biti 1–39."
                )

            if len(set(numbers)) != PICKS:
                raise ValueError(
                    f"Red {line_number}: broj se ponavlja."
                )

            # Sortiraju se samo brojevi unutar izvlačenja.
            # Redosled istorijskih redova ostaje nepromenjen.
            rows.append(sorted(numbers))

    required = (
        WINDOW
        + MIN_TRAIN_SAMPLES
        + VALIDATION_SIZE
        + TEST_SIZE
    )

    if len(rows) < required:
        raise ValueError(
            f"Potrebno najmanje {required} izvlačenja; "
            f"pronađeno {len(rows)}."
        )

    return path, np.asarray(rows, dtype=np.int64)


def prepare_data(draws):
    # Fiksna promena skale; nema procene iz budućih podataka.
    scaled = draws.astype(np.float32) / float(BALLS)

    count = len(draws) - WINDOW
    inputs = np.empty(
        (count, WINDOW, PICKS),
        dtype=np.float32,
    )

    for sample in range(count):
        inputs[sample] = scaled[sample:sample + WINDOW]

    outcomes = draws[WINDOW:].copy()
    grid = np.arange(1, BALLS + 1)

    # Regresioni ciljevi: empirijske kumulativne raspodele
    # sedam sortiranih pozicija.
    targets = (
        outcomes[:, :, None] <= grid[None, None, :]
    ).astype(np.float64)

    # Pravi NEXT prozor uključuje poslednje poznato kolo.
    next_input = scaled[-WINDOW:][None].copy()

    return inputs, targets, outcomes, next_input


def analytic_fill(parameter, phase):
    """Analitička inicijalizacija bez slučajnih brojeva."""
    with torch.no_grad():
        if parameter.ndim == 1:
            parameter.zero_()
            return

        rows = parameter.shape[0]
        columns = parameter.numel() // rows

        row = torch.arange(
            1,
            rows + 1,
            dtype=torch.float64,
            device=parameter.device,
        )[:, None]

        column = torch.arange(
            1,
            columns + 1,
            dtype=torch.float64,
            device=parameter.device,
        )[None, :]

        angles = (
            math.sqrt(2.0) * row * column
            + 0.113 * row.square()
            + 0.071 * column.square()
            + phase
        )

        values = torch.sin(angles) * math.sqrt(2.0 / columns)

        parameter.copy_(
            values.reshape(parameter.shape).to(parameter.dtype)
        )


class AnalyticConv1d(nn.Conv1d):
    def __init__(self, input_channels, output_channels, phase):
        self.phase = float(phase)

        super().__init__(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=True,
        )

    def reset_parameters(self):
        analytic_fill(self.weight, self.phase)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()


class AnalyticLinear(nn.Linear):
    def reset_parameters(self):
        analytic_fill(self.weight, phase=2.83)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()


class AnalyticLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, phase):
        self.phase = float(phase)

        # Konstruktor poziva redefinisani reset_parameters.
        # Nema prethodne slučajne inicijalizacije.
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

    def reset_parameters(self):
        for index, (_, parameter) in enumerate(
            self.named_parameters(recurse=False)
        ):
            analytic_fill(
                parameter,
                self.phase + 0.379 * index,
            )

        with torch.no_grad():
            # PyTorch redosled kapija: i, f, g, o.
            self.bias_ih_l0[
                self.hidden_size:2 * self.hidden_size
            ].fill_(1.0)


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = AnalyticConv1d(
            PICKS, 64, phase=0.19
        )
        self.pool_1 = nn.MaxPool1d(kernel_size=2)

        self.conv_2 = AnalyticConv1d(
            64, 32, phase=0.73
        )
        self.pool_2 = nn.MaxPool1d(kernel_size=2)

        self.lstm_1 = AnalyticLSTM(
            32, 64, phase=1.31
        )
        self.lstm_2 = AnalyticLSTM(
            64, 32, phase=1.97
        )

        self.output = AnalyticLinear(
            32, PICKS * BALLS
        )

        rank = torch.arange(PICKS)[:, None]
        index = torch.arange(BALLS)[None, :]

        # Podrška proizlazi samo iz sortiranja sedam
        # različitih brojeva iz raspona 1–39.
        legal_support = (
            (index >= rank)
            & (index <= BALLS - PICKS + rank)
        )

        self.register_buffer("legal_support", legal_support)

    def forward(self, inputs):
        # (B, 30, 7) -> (B, 7, 30)
        values = inputs.transpose(1, 2).contiguous()

        # Vremenska dužina: 30 -> 28 -> 14 -> 12 -> 6.
        values = self.pool_1(F.relu(self.conv_1(values)))
        values = self.pool_2(F.relu(self.conv_2(values)))

        # (B, 32, 6) -> (B, 6, 32)
        values = values.transpose(1, 2).contiguous()

        sequence, _ = self.lstm_1(values)
        _, (hidden, _) = self.lstm_2(sequence)

        logits = self.output(hidden[-1]).reshape(
            -1, PICKS, BALLS
        ).to(torch.float64)

        # Pozitivne mase, učene regresijom CDF.
        mass = F.softplus(logits) + 1e-12
        mass = mass * self.legal_support[None]

        return mass / mass.sum(dim=-1, keepdim=True)


def input_tensor(values):
    return torch.from_numpy(
        np.ascontiguousarray(values, dtype=np.float32)
    ).to(DEVICE)


def target_tensor(values):
    return torch.from_numpy(
        np.ascontiguousarray(values, dtype=np.float64)
    ).to(DEVICE)


def regression_loss(mass, target_cdf):
    return (
        mass.cumsum(dim=-1) - target_cdf
    ).square().mean()


def train_epoch(model, optimizer, inputs, targets):
    model.train()

    total_error = 0.0
    total_samples = 0

    # Bez mešanja redosleda trening uzoraka.
    for start in range(0, len(inputs), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(inputs))

        optimizer.zero_grad(set_to_none=True)

        mass = model(inputs[start:stop])
        loss = regression_loss(mass, targets[start:stop])

        if not torch.isfinite(loss):
            raise RuntimeError("Nekonačna greška pri treniranju.")

        loss.backward()

        norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        if not torch.isfinite(norm):
            raise RuntimeError("Nekonačan gradijent.")

        optimizer.step()

        total_error += float(loss.detach()) * (stop - start)
        total_samples += stop - start

    return total_error / total_samples


def select_epochs(train_x, train_y, validation_x, validation_y):
    model = CNNLSTM().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_inputs = input_tensor(train_x)
    train_targets = target_tensor(train_y)

    validation_inputs = input_tensor(validation_x)
    validation_targets = target_tensor(validation_y)

    best_epoch = 0
    best_error = float("inf")
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_error = train_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
        )

        model.eval()

        with torch.no_grad():
            validation_error = float(
                regression_loss(
                    model(validation_inputs),
                    validation_targets,
                )
            )

        if not math.isfinite(validation_error):
            raise RuntimeError("Nekonačna validaciona greška.")

        if validation_error < best_error:
            best_epoch = epoch
            best_error = validation_error
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Izbor epoha {epoch}/{MAX_EPOCHS} | "
                f"trening={train_error:.6f} | "
                f"validacija={validation_error:.6f}",
                flush=True,
            )

        if stale >= PATIENCE:
            break

    if best_epoch < 1:
        raise RuntimeError("Izbor broja epoha nije uspeo.")

    return best_epoch, best_error


def fit_fixed(inputs, targets, epochs, label):
    model = CNNLSTM().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    tensor_x = input_tensor(inputs)
    tensor_y = target_tensor(targets)

    for epoch in range(1, epochs + 1):
        error = train_epoch(
            model,
            optimizer,
            tensor_x,
            tensor_y,
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  {label}: epoha {epoch}/{epochs} | "
                f"greška={error:.6f}",
                flush=True,
            )

    return model


def predict_masses(model, inputs):
    model.eval()
    blocks = []

    with torch.no_grad():
        for start in range(0, len(inputs), BATCH_SIZE):
            block = input_tensor(
                inputs[start:start + BATCH_SIZE]
            )
            blocks.append(model(block).cpu().numpy())

    masses = np.concatenate(blocks, axis=0)

    if not np.isfinite(masses).all():
        raise RuntimeError("Neispravan numerički izlaz modela.")

    return masses


def select_combination(mass):
    """
    Tačno dinamičko programiranje za proizvod naučenih
    marginalnih masa sedam sortiranih pozicija.

    Maksimizuje:
        sum_j log(mass[j, n_j - 1])

    uz jedina ograničenja:
        1 <= n1 < n2 < ... < n7 <= 39.

    Bez zaokruživanja, pomeranja duplikata i dodatnih filtera.
    """
    if mass.shape != (PICKS, BALLS):
        raise ValueError("Pogrešan oblik raspodele.")

    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("Neispravne mase raspodele.")

    log_mass = np.full(
        (PICKS, BALLS),
        -np.inf,
        dtype=np.float64,
    )
    np.log(mass, out=log_mass, where=mass > 0)

    scores = np.full(
        (PICKS, BALLS),
        -np.inf,
        dtype=np.float64,
    )
    parents = np.full(
        (PICKS, BALLS),
        -1,
        dtype=np.int64,
    )

    scores[0, :BALLS - PICKS + 1] = (
        log_mass[0, :BALLS - PICKS + 1]
    )

    for rank in range(1, PICKS):
        best_score = -np.inf
        best_index = -1

        for number_index in range(BALLS):
            previous = number_index - 1

            if (
                previous >= 0
                and scores[rank - 1, previous] > best_score
            ):
                best_score = scores[rank - 1, previous]
                best_index = previous

            if not (
                rank <= number_index <= BALLS - PICKS + rank
            ):
                continue

            if best_index >= 0:
                scores[rank, number_index] = (
                    best_score + log_mass[rank, number_index]
                )
                parents[rank, number_index] = best_index

    current = int(np.argmax(scores[-1]))

    if not np.isfinite(scores[-1, current]):
        raise RuntimeError("Nema važeće kombinacije.")

    selected = [0] * PICKS
    selected[-1] = current + 1

    for rank in range(PICKS - 1, 0, -1):
        current = int(parents[rank, current])

        if current < 0:
            raise RuntimeError("Neispravna putanja izbora.")

        selected[rank - 1] = current + 1

    combination = tuple(selected)

    if (
        len(set(combination)) != PICKS
        or combination[0] < 1
        or combination[-1] > BALLS
        or combination != tuple(sorted(combination))
    ):
        raise RuntimeError("Izlaz nije važeća Loto kombinacija.")

    return combination


def numbers_text(numbers):
    return ", ".join(str(int(number)) for number in numbers)


def longest_run(numbers):
    best = current = 1

    for previous, following in zip(numbers[:-1], numbers[1:]):
        current = current + 1 if following == previous + 1 else 1
        best = max(best, current)

    return best


def report_test(masses, targets, outcomes, reference_cdf):
    predicted_cdf = masses.cumsum(axis=-1)

    model_error = float(
        np.mean((predicted_cdf - targets) ** 2)
    )
    reference_error = float(
        np.mean((reference_cdf[None] - targets) ** 2)
    )

    hits = []

    for mass, actual in zip(masses, outcomes):
        prediction = select_combination(mass)
        hits.append(
            len(set(prediction).intersection(actual.tolist()))
        )

    counts = np.bincount(hits, minlength=PICKS + 1)

    print(f"  Provereno kola: {len(hits)}")
    print(f"  CDF regresiona greška: {model_error:.6f}")
    print(
        "  CDF greška istorijskog proseka, za poređenje: "
        f"{reference_error:.6f}"
    )
    print(f"  Prosečno pogodaka: {np.mean(hits):.4f}")
    print(
        "  Pogoci 0–7: "
        + " | ".join(
            f"{number}: {int(counts[number])}"
            for number in range(PICKS + 1)
        )
    )


def process_file(filename):
    path, draws = read_draws(filename)
    inputs, targets, outcomes, next_input = prepare_data(draws)

    train_end = (
        len(inputs) - VALIDATION_SIZE - TEST_SIZE
    )
    validation_end = len(inputs) - TEST_SIZE

    fingerprint = hashlib.sha256(
        draws.astype("<i8", copy=False).tobytes()
    ).hexdigest()[:16]

    print("\n" + "=" * 72)
    print(path.name)
    print(
        f"Izvlačenja: {len(draws)} | "
        f"istorijski prelazi: {len(inputs)} | "
        f"oznaka sadržaja: {fingerprint}",
        flush=True,
    )

    print("\n1. Hronološki izbor broja epoha")

    epochs, validation_error = select_epochs(
        inputs[:train_end],
        targets[:train_end],
        inputs[train_end:validation_end],
        targets[train_end:validation_end],
    )

    print(
        f"  Izabrano epoha: {epochs} | "
        f"validaciona greška: {validation_error:.6f}"
    )

    print("\n2. Završna provera na izdvojenim kolima")

    test_model = fit_fixed(
        inputs[:validation_end],
        targets[:validation_end],
        epochs,
        label="Model za proveru",
    )

    # Parametri modela su fiksirani pre završnog testa.
    # Ulaz svakog test kola sadrži samo prethodno poznata kola.
    test_masses = predict_masses(
        test_model,
        inputs[validation_end:],
    )

    reference_cdf = targets[:validation_end].mean(axis=0)

    report_test(
        test_masses,
        targets[validation_end:],
        outcomes[validation_end:],
        reference_cdf,
    )

    del test_model, test_masses

    print(
        "\n3. Završno treniranje na celoj poznatoj istoriji",
        flush=True,
    )

    final_model = fit_fixed(
        inputs,
        targets,
        epochs,
        label="Završni model",
    )

    next_mass = predict_masses(
        final_model,
        next_input,
    )[0]

    combination = select_combination(next_mass)

    print("\nKontrola: prethodno izvuceno kolo")
    print(numbers_text(draws[-1]))

    print("\nNEXT:")
    print(numbers_text(combination))

    low = sum(number <= 19 for number in combination)
    historical_copies = int(
        np.sum(
            np.all(
                draws == np.asarray(combination)[None],
                axis=1,
            )
        )
    )

    print(
        "\nOpis izlaza: "
        f"najduži uzastopni niz={longest_run(combination)}; "
        f"1–19: {low}; 20–39: {PICKS - low}; "
        f"ista kombinacija u istoriji: {historical_copies}."
    )
    print("Opis izlaza ne menja izabranu kombinaciju.")

    del final_model

    return path.name, combination


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Loto 7/39 — CNN–LSTM V1."
    )

    parser.add_argument(
        "csv_files",
        nargs="*",
        help="Tri CSV putanje; bez argumenata koristi DEFAULT_FILES.",
    )

    args = parser.parse_args()

    if args.csv_files and len(args.csv_files) != 3:
        parser.error("Potrebno je proslediti tačno tri CSV putanje.")

    return args.csv_files or list(DEFAULT_FILES)


def main():
    filenames = parse_arguments()
    configure_torch()

    print("LOTO 7/39 — CNN–LSTM V1")
    print(
        "Konvolucioni slojevi -> vremenski LSTM slojevi "
        "-> regresija raspodela."
    )

    results = []
    failures = []

    for filename in filenames:
        try:
            results.append(process_file(filename))
        except (ValueError, OSError, RuntimeError, csv.Error) as exc:
            failures.append((str(filename), str(exc)))
            print(
                f"\nGREŠKA za {Path(filename).name}: {exc}",
                flush=True,
            )

    if results:
        print("\n" + "=" * 72)
        print("KONAČNI NEXT REZULTATI")

        for name, combination in results:
            print(f"{name}: ({numbers_text(combination)})")

        for index, (first_name, first) in enumerate(results):
            for second_name, second in results[index + 1:]:
                if first == second:
                    print(
                        f"\nIsti NEXT: {first_name} / {second_name}: "
                        f"({numbers_text(first)}). "
                        "Rezultat nije naknadno menjan."
                    )

    if failures:
        print(
            "\nZa neuspešno obrađene fajlove "
            "nema zamenskih kombinacija."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())



"""
LOTO 7/39 — CNN-LSTM V1
Konvolucioni slojevi -> vremenski LSTM slojevi -> regresija raspodela.


========================================================================
loto7_4704_k83.csv
Izvlačenja: 4704 | istorijski prelazi: 4674 | oznaka sadržaja: 1badcf2632a9f966

1. Hronološki izbor broja epoha
  Izbor epoha 1/80 | trening=0.113387 | validacija=0.084136
  Izbor epoha 10/80 | trening=0.073267 | validacija=0.069628
  Izbor epoha 20/80 | trening=0.073193 | validacija=0.069498
  Izbor epoha 30/80 | trening=0.073178 | validacija=0.069470
  Izbor epoha 40/80 | trening=0.073172 | validacija=0.069463
  Izbor epoha 50/80 | trening=0.073169 | validacija=0.069461
  Izbor epoha 60/80 | trening=0.073168 | validacija=0.069460
  Izbor epoha 70/80 | trening=0.073167 | validacija=0.069459
  Izbor epoha 80/80 | trening=0.073167 | validacija=0.069459
  Izabrano epoha: 77 | validaciona greška: 0.069459

2. Završna provera na izdvojenim kolima
  Model za proveru: epoha 1/77 | greška=0.113092
  Model za proveru: epoha 10/77 | greška=0.073260
  Model za proveru: epoha 20/77 | greška=0.073180
  Model za proveru: epoha 30/77 | greška=0.073163
  Model za proveru: epoha 40/77 | greška=0.073156
  Model za proveru: epoha 50/77 | greška=0.073153
  Model za proveru: epoha 60/77 | greška=0.073151
  Model za proveru: epoha 70/77 | greška=0.073150
  Model za proveru: epoha 77/77 | greška=0.073150
  Provereno kola: 64
  CDF regresiona greška: 0.073840
  CDF greška istorijskog proseka, za poređenje: 0.073849
  Prosečno pogodaka: 1.2656
  Pogoci 0-7: 0: 19 | 1: 22 | 2: 14 | 3: 5 | 4: 4 | 5: 0 | 6: 0 | 7: 0

3. Završno treniranje na celoj poznatoj istoriji
  Završni model: epoha 1/77 | greška=0.112717
  Završni model: epoha 10/77 | greška=0.073233
  Završni model: epoha 20/77 | greška=0.073161
  Završni model: epoha 30/77 | greška=0.073146
  Završni model: epoha 40/77 | greška=0.073141
  Završni model: epoha 50/77 | greška=0.073138
  Završni model: epoha 60/77 | greška=0.073137
  Završni model: epoha 70/77 | greška=0.073136
  Završni model: epoha 77/77 | greška=0.073135

Kontrola: prethodno izvuceno kolo
7, 10, 13, 27, 28, 31, 38

NEXT:
1, x, 12, y, 29, z, 39

Opis izlaza: najduži uzastopni niz=1; 1-19: 3; 20-39: 4; ista kombinacija u istoriji: 0.
Opis izlaza ne menja izabranu kombinaciju.



========================================================================
loto7_4704_k83_loto_2974.csv
Izvlačenja: 2974 | istorijski prelazi: 2944 | oznaka sadržaja: 992bc400ed1af4f4

1. Hronološki izbor broja epoha
  Izbor epoha 1/80 | trening=0.124695 | validacija=0.101003
  Izbor epoha 10/80 | trening=0.073357 | validacija=0.069984
  Izbor epoha 20/80 | trening=0.073203 | validacija=0.069878
  Izbor epoha 30/80 | trening=0.073170 | validacija=0.069863
  Izbor epoha 40/80 | trening=0.073157 | validacija=0.069852
  Izbor epoha 50/80 | trening=0.073150 | validacija=0.069844
  Izbor epoha 60/80 | trening=0.073146 | validacija=0.069840
  Izbor epoha 70/80 | trening=0.073144 | validacija=0.069837
  Izbor epoha 80/80 | trening=0.073142 | validacija=0.069835
  Izabrano epoha: 80 | validaciona greška: 0.069835

2. Završna provera na izdvojenim kolima
  Model za proveru: epoha 1/80 | greška=0.124331
  Model za proveru: epoha 10/80 | greška=0.073289
  Model za proveru: epoha 20/80 | greška=0.073140
  Model za proveru: epoha 30/80 | greška=0.073107
  Model za proveru: epoha 40/80 | greška=0.073094
  Model za proveru: epoha 50/80 | greška=0.073088
  Model za proveru: epoha 60/80 | greška=0.073084
  Model za proveru: epoha 70/80 | greška=0.073082
  Model za proveru: epoha 80/80 | greška=0.073080
  Provereno kola: 64
  CDF regresiona greška: 0.072465
  CDF greška istorijskog proseka, za poređenje: 0.072466
  Prosečno pogodaka: 1.3125
  Pogoci 0-7: 0: 15 | 1: 23 | 2: 20 | 3: 3 | 4: 3 | 5: 0 | 6: 0 | 7: 0

3. Završno treniranje na celoj poznatoj istoriji
  Završni model: epoha 1/80 | greška=0.123882
  Završni model: epoha 10/80 | greška=0.073275
  Završni model: epoha 20/80 | greška=0.073125
  Završni model: epoha 30/80 | greška=0.073093
  Završni model: epoha 40/80 | greška=0.073080
  Završni model: epoha 50/80 | greška=0.073073
  Završni model: epoha 60/80 | greška=0.073070
  Završni model: epoha 70/80 | greška=0.073067
  Završni model: epoha 80/80 | greška=0.073066

Kontrola: prethodno izvuceno kolo
9, x, 17, y, 24, z, 35

NEXT:
1, 8, 13, 21, 29, 33, 39

Opis izlaza: najduži uzastopni niz=1; 1-19: 3; 20-39: 4; ista kombinacija u istoriji: 0.
Opis izlaza ne menja izabranu kombinaciju.



========================================================================
loto7_4704_k83_loto_plus_1730.csv
Izvlačenja: 1730 | istorijski prelazi: 1700 | oznaka sadržaja: fc57041695a80b3c

1. Hronološki izbor broja epoha
  Izbor epoha 1/80 | trening=0.133640 | validacija=0.125699
  Izbor epoha 10/80 | trening=0.073662 | validacija=0.076280
  Izbor epoha 20/80 | trening=0.073340 | validacija=0.075931
  Izbor epoha 30/80 | trening=0.073251 | validacija=0.075846
  Izbor epoha 40/80 | trening=0.073214 | validacija=0.075816
  Izbor epoha 50/80 | trening=0.073196 | validacija=0.075802
  Izbor epoha 60/80 | trening=0.073186 | validacija=0.075793
  Izbor epoha 70/80 | trening=0.073180 | validacija=0.075787
  Izbor epoha 80/80 | trening=0.073175 | validacija=0.075784
  Izabrano epoha: 80 | validaciona greška: 0.075784

2. Završna provera na izdvojenim kolima
  Model za proveru: epoha 1/80 | greška=0.133462
  Model za proveru: epoha 10/80 | greška=0.073734
  Model za proveru: epoha 20/80 | greška=0.073413
  Model za proveru: epoha 30/80 | greška=0.073323
  Model za proveru: epoha 40/80 | greška=0.073287
  Model za proveru: epoha 50/80 | greška=0.073269
  Model za proveru: epoha 60/80 | greška=0.073259
  Model za proveru: epoha 70/80 | greška=0.073253
  Model za proveru: epoha 80/80 | greška=0.073249
  Provereno kola: 64
  CDF regresiona greška: 0.072664
  CDF greška istorijskog proseka, za poređenje: 0.072629
  Prosečno pogodaka: 1.3750
  Pogoci 0-7: 0: 7 | 1: 35 | 2: 14 | 3: 7 | 4: 1 | 5: 0 | 6: 0 | 7: 0

3. Završno treniranje na celoj poznatoj istoriji
  Završni model: epoha 1/80 | greška=0.133035
  Završni model: epoha 10/80 | greška=0.073684
  Završni model: epoha 20/80 | greška=0.073384
  Završni model: epoha 30/80 | greška=0.073297
  Završni model: epoha 40/80 | greška=0.073265
  Završni model: epoha 50/80 | greška=0.073249
  Završni model: epoha 60/80 | greška=0.073241
  Završni model: epoha 70/80 | greška=0.073235
  Završni model: epoha 80/80 | greška=0.073232

Kontrola: prethodno izvuceno kolo
7, 10, 13, 27, 28, 31, 38

NEXT:
1, x, 12, y, 27, z, 39

Opis izlaza: najduži uzastopni niz=1; 1–19: 3; 20–39: 4; ista kombinacija u istoriji: 0.
Opis izlaza ne menja izabranu kombinaciju.


========================================================================
KONAČNI NEXT REZULTATI
loto7_4704_k83.csv: (1, x, 12, y, 29, z, 39)
loto7_4704_k83_loto_2974.csv: (1, x, 13, y, 29, z, 39)
loto7_4704_k83_loto_plus_1730.csv: (1, x, 12, y, 27, z, 39)
"""



"""
povezani CNN-LSTM 
U glavnom trening-kodu konvolucioni slojevi prvo izdvajaju lokalne obrasce iz istorijskog prozora. 
Njihov izlaz zatim prolazi kroz dva LSTM sloja, koji uče vremenske veze između tih obrazaca. 
Cela mreža trenira se zajedno.
Konvoluciona mreža priprema informacije koje LSTM dalje obrađuje.
Regresija raspodela za 7/39, celu istoriju, odvojene modele i hronološku proveru.
"""
