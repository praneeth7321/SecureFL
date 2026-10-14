# SecureFL — Detecting Backdoor Attacks in Federated Learning

SecureFL is a student project prototype that demonstrates a backdoor attack in a federated learning simulation and explores a Krum-style robust aggregation defense.

## Project Overview

Federated learning lets multiple clients train a shared model while keeping their raw training data local. However, a malicious client may poison its local training data so that inputs containing a trigger are classified as a chosen target label.

SecureFL simulates five clients:
- Four intended honest clients
- One configured malicious client
- A white-square trigger used for backdoor poisoning
- A server-side Krum-style update-selection step before aggregation

The prototype evaluates model performance using clean accuracy and Attack Success Rate (ASR) across five communication rounds.

> **Research note:** This is a prototype. Krum-style selection is a heuristic in this implementation, and a low ASR by itself does not prove the defense caused the result. Compare against an undefended baseline before making effectiveness claims.

## Features

- Federated learning simulation using Flower
- CNN training on MNIST
- Five simulated clients
- Trigger-based data poisoning by a malicious client
- Krum-style scoring and update selection
- Clean accuracy and ASR reporting
- Round-by-round logs and generated graphs
- Streamlit dashboard for presenting results

## Project Structure

```text
SecureFL/
├── aggregation/
│   └── secure_aggregation.py
├── attacks/
├── clients/
│   ├── client.py
│   └── malicious_client.py
├── datasets/
│   └── load_data.py
├── detection/
│   └── krum.py
├── evaluation/
├── graphs/
├── models/
│   └── cnn_model.py
├── server/
│   └── server.py
├── securefl/
│   ├── client_app.py
│   └── server_app.py
├── dashboard.py
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

*The exact files may vary as the project evolves.*

## Requirements

- Windows, macOS, or Linux
- Python 3.11 recommended for the current environment
- Git (optional, for version control)
- Internet connection for installing dependencies and downloading MNIST on first run

## Setup on Windows (PowerShell)

Open a terminal in the project root (the folder containing `main.py`).

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### 2. Install project dependencies

Activate the environment if PowerShell permits it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, you can use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Install dashboard dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit pandas altair
```

## Run the simulation

From the project root:

```powershell
.\.venv\Scripts\python.exe main.py
```

The existing `main.py` path uses Flower's legacy `start_simulation()` API in the current code and may display deprecation warnings. The simulation can still complete in the current environment.

A successful run should show:
- Five communication rounds
- Selected and excluded update identifiers
- Clean accuracy and Attack Success Rate after rounds
- A final training-completed message
- Graphs saved in the `graphs/` folder, if graph generation completes

Do not assume the malicious client was correctly excluded in every round merely because one update was excluded. Check the logs and validate against a baseline.

## Run the dashboard

The Streamlit dashboard provides a presentation interface for the simulation logs and metrics.

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

Dashboard functions may include:
- Starting the simulation from the UI
- Showing parsed clean accuracy and ASR by round
- Displaying selected/excluded update identifiers
- Showing generated graphs
- Loading a previously saved log

If the metrics are blank, confirm a log is loaded and that it contains lines like `Clean Accuracy: ...%` and `Attack Success Rate: ...%`.

## Metrics

- **Clean Accuracy:** fraction of ordinary test examples predicted correctly, expressed as a percentage.
- **Attack Success Rate (ASR):** fraction of triggered test examples classified as the attacker's target label, expressed as a percentage.
- **Krum-style update selection:** scores client updates using distances and selects a subset for aggregation. The implementation should be described as Krum-style unless it has been verified to match the original Krum or Multi-Krum algorithm precisely.

## Latest example results

Replace this section with the metrics from the exact run you are presenting. The latest dashboard log shared during project preparation reported:

| Round | Clean accuracy | Attack success rate |
|---:|---:|---:|
| 1 | 96.43% | 0.41% |
| 2 | 98.55% | 0.21% |
| 3 | 98.94% | 0.59% |
| 4 | 98.99% | 0.28% |
| 5 | 99.16% | 1.45% |

These are example results from one run, not guaranteed reproducible values. Training results can vary between runs.

## Known limitations and next steps

- The current Flower simulation path may emit deprecation warnings.
- Krum-style filtering does not guarantee that the malicious client is detected in every round.
- Update identifiers printed by Flower may not directly identify human-readable client numbers.
- Reported metrics should be checked against a run without the defense.
- Future work: run an undefended FedAvg baseline, repeat experiments with fixed seeds, vary the malicious-client ratio and poisoning rate, and report mean and standard deviation over multiple runs.

## References to verify and include in the final report

1. McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
2. Blanchard, P. et al. (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.* NeurIPS.
3. Bagdasaryan, E. et al. (2020). *How To Backdoor Federated Learning.* AISTATS.

Verify citation details and format them according to your course requirements before submission.

## Team

- Team members: [Add names and IDs]
- Faculty mentor: [Add name]
- Course: DSE3170 — Project Based Learning-3
- Institution: Manipal University Jaipur
