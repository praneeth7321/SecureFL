# SecureFL: Federated IoT intrusion detection

SecureFL uses Flower to train a binary intrusion detector on **TON_IoT** and **CICIoT2023** separately. Each run partitions traffic among five simulated clients. Four clients train on clean labels; client 4 flips a fraction of attack labels to benign. The server scores model updates with Krum, excludes the least consistent update, aggregates the other four, and evaluates on held-out traffic. This is a label-poisoning experiment, not an image-trigger backdoor experiment. The attacker’s effect is measured as the attack miss rate, alongside accuracy, precision, recall, and F1.

## Data

The supplied TON_IoT network CSV goes in `data/ton_iot/`. The CICIoT2023 Kaggle mirror used for validation is https://www.kaggle.com/datasets/himadri07/ciciot2023; its CSV files go in `data/ciciot2023/train/train.csv` and `data/ciciot2023/test/test.csv` (the supplied validation file is optional). Dataset files are ignored by Git. TON_IoT CSVs need a binary `label` column containing `0`/`1` or benign/attack. CICIoT2023 CSVs need a `label` column where `BenignTraffic` is benign and every other label is an attack. The loader uses numeric traffic features, excludes identifiers and known label columns, and samples up to 5,000 rows per class for training. CICIoT2023 uses its supplied train/test split and a held-out test sample of up to 1,000 rows per class. TON_IoT uses a stratified 80/20 split after sampling. Missing-value imputation and quantile scaling are fitted only on training records. Each dataset gets its own model because their feature sets differ.

For the datasets validated in this workspace, the TON_IoT CSV has SHA-256 `26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974` and the Kaggle ZIP has SHA-256 `793b212d1dc40973068d06cec4d0676eaf7f17436a859dfa522303e199c7ea5a`. These identify the exact files used for the reported results.

**Check your source export before running:** numeric codes that encode attack labels under other column names must be removed from the input files. The loader cannot identify arbitrary label leakage. For a reproducible run, keep a fixed export and record its source and checksum externally.

To prepare the files on your own computer, use the two ZIP archives supplied for this project:

```bash
python3.12 scripts/prepare_data.py --ton-archive /path/to/archive.zip --cic-archive "/path/to/archive(1).zip"
```

If you do not have the second ZIP locally, omit `--cic-archive` to download the public Kaggle copy. This step needs about 3 GB of free disk space.

## Install and run

Python 3.12 was used for validation. With `uv` installed, create a CPU environment outside the checkout:

```bash
uv venv /workspace/securefl-env/venv --python python3.12
uv pip install --python /workspace/securefl-env/venv/bin/python --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple --index-strategy unsafe-best-match -r requirements.txt
```

From the project root:

```bash
MPLBACKEND=Agg /workspace/securefl-env/venv/bin/python main.py --data-dir data --output-dir results
```

Use `--datasets ton_iot` or `--datasets ciciot2023` to run one dataset, and `--rounds`, `--local-epochs`, `--max-rows-per-class`, or `--seed` to change experiment size and sampling. The default trains each client for three local epochs per round. The default runs both datasets for five rounds. Ray starts locally through Flower; no dashboard or web service is used.

Each dataset writes `round_metrics.csv`, `classification_metrics.png`, and `attack_miss_rate.png` under its own output directory. The output root also contains `dataset_summary.csv` and `dataset_comparison.png` for the final-round comparison. Tables are printed to the terminal. Accuracy alone can hide missed attacks; review recall, F1, and attack miss rate together. The rejected Flower proxy ID is reported for diagnostics, but it is not a reliable proof of which logical client was malicious.

## Windows PowerShell quick start

Install Git and Python 3.12 first. The commands below install `uv` through Python, so `uv` does not need to be on your PowerShell PATH. Keep the two uploaded ZIP files in your Downloads folder, or adjust the paths below. Allow at least 5 GB of free disk space.

```powershell
git clone https://github.com/praneeth7321/SecureFL.git
cd SecureFL
py -3.12 scripts\prepare_data.py --ton-archive "$HOME\Downloads\archive.zip" --cic-archive "$HOME\Downloads\archive(1).zip"
py -3.12 -m pip install --user uv
py -3.12 -m uv venv .venv --python 3.12
py -3.12 -m uv pip install --python .venv\Scripts\python.exe --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple --index-strategy unsafe-best-match -r requirements.txt
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python.exe main.py --data-dir data --output-dir results
```

The run is finished when both five-round tables are printed. Open `results\dataset_summary.csv` and `results\dataset_comparison.png` for the overall comparison; each dataset also has its own CSV and graphs. If you already cloned the repository, use `git pull` in that checkout instead of cloning it again. If the second ZIP is missing, omit `--cic-archive`; the preparation script downloads the public Kaggle copy.
