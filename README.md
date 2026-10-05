# SecureFL: Detecting Backdoor Attacks using Federated Learning

## Project Overview

SecureFL is a command-line Python application that demonstrates how a backdoor attack can affect a Federated Learning (FL) system and how Krum-based robust aggregation can reduce the impact of an unusual client update.

The project trains one shared Convolutional Neural Network (CNN) on the MNIST handwritten-digit dataset through Flower. Five simulated clients participate in every training round: four honest clients and one malicious client. The malicious client inserts a small white-square trigger into some of its local MNIST images and changes their labels to a configurable target digit.

The server applies Krum scoring to the received client model updates, rejects the least-consistent update for that round, averages the remaining updates, and evaluates the updated global model.

## Problem Statement

Federated Learning allows multiple clients to train a shared model without pooling their raw data on one central server. However, a malicious participant can poison its local training data so that the global model behaves normally on clean data but predicts an attacker-chosen label when a visual trigger is present.

This project studies that targeted backdoor threat on MNIST and demonstrates a simple Krum-based robust aggregation defense.

## Objectives

- Simulate Federated Learning with Flower 1.33 and five clients.
- Train a simple PyTorch CNN on client-partitioned MNIST data.
- Create a targeted backdoor attack using a small white-square image trigger.
- Use Krum to score client model updates by their consistency with other updates.
- Reject the least-consistent update and aggregate the remaining accepted updates.
- Measure clean test accuracy, training loss, attack success rate (ASR), and the rejected client ID.
- Save accuracy and ASR graphs for each training run.

## System Architecture

```text
                         +-----------------------+
                         |    Flower Server      |
                         | Global CNN + Krum +   |
                         | Filtered Aggregation  |
                         +-----------+-----------+
                                     |
                         Global CNN model weights
                                     |
          +--------------------------+--------------------------+
          |             |             |             |          |
          v             v             v             v          v
     Client 0      Client 1      Client 2      Client 3     Client 4
      Honest        Honest        Honest        Honest      Malicious
          |             |             |             |          |
      Clean MNIST   Clean MNIST   Clean MNIST   Clean MNIST  Clean + poisoned
          |             |             |             |       MNIST with trigger
          +-------------+-------------+-------------+----------+
                                     |
                         Local CNN model updates
                                     |
                         +-----------v-----------+
                         | Krum scores updates    |
                         | Reject one update      |
                         | Average accepted four  |
                         +-----------+-----------+
                                     |
                              Updated global CNN
                                     |
                    Clean accuracy, loss, ASR, graphs
```

## Folder Structure

```text
SecureFL/
├── models/
├── datasets/
├── clients/
├── attacks/
├── detection/
├── aggregation/
├── server/
├── evaluation/
├── graphs/
│   ├── accuracy_vs_round.png
│   └── asr_vs_round.png
├── main.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python 3.14
- Flower 1.33 for Federated Learning simulation
- PyTorch for CNN training
- torchvision MNIST for dataset download and preprocessing
- NumPy for numerical update processing
- Pandas for round-level experiment records
- Scikit-learn for permitted evaluation support
- Matplotlib for accuracy and ASR graphs

## Installation Steps

1. Open a terminal in the `SecureFL` project folder.
2. Confirm that Python 3.14 is available:

   ```powershell
   python --version
   ```

3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the pinned project dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## How to Run

Run the project from the root `SecureFL` directory:

```powershell
python main.py
```

The first run downloads the MNIST dataset through torchvision. The program then runs the configured number of federated training rounds and saves graphs in the `graphs/` folder.

## Workflow

```text
Initialize Flower Server
        ↓
Create 5 Clients
        ↓
Distribute Global CNN
        ↓
Honest Clients Train
        ↓
Malicious Client Injects Trigger
        ↓
Clients Send Updates
        ↓
Server Runs Krum
        ↓
Reject Suspicious Update
        ↓
Aggregate Remaining Updates
        ↓
Update Global Model
        ↓
Evaluate Accuracy & ASR
        ↓
Save Graphs
```

## Krum Explanation

Krum is a robust aggregation algorithm, not a malicious-client detector. It computes scores for client updates using pairwise distances and selects updates that are most consistent with the majority.

In this project, the server calculates a Krum score for each of the five submitted model updates. The update with the least-consistent score is rejected before aggregation, while the other four updates are averaged. For evaluation purposes, the rejected update is treated as the suspected malicious update and compared with the known malicious client, Client 4.

Therefore, a rejected Client 4 is a successful defensive indication in this controlled simulation, but Krum itself does not prove that a client is malicious.

## Backdoor Attack Explanation

The malicious client, Client 4, performs a targeted data-poisoning attack during its local training:

1. It selects a configurable fraction of its MNIST training images.
2. It adds a small white square to a fixed corner of each selected image.
3. It changes each selected image label to a configurable target class, such as digit `0`.
4. It trains on a mixture of clean and poisoned images.

The goal is for the global model to retain useful clean-digit accuracy while classifying triggered images as the target label.

## Robust Aggregation using Krum

The server uses Krum scores to identify the update that is least consistent with the other client updates in the current round. It rejects that selected update and averages only the remaining four accepted updates to produce the next global model.

This is a simplified robust aggregation process intended to limit the influence of an unusual update. It does not use encryption or Flower SecAgg/SecAgg+.

## Evaluation Metrics

- **Clean Accuracy:** Percentage of clean MNIST test images classified correctly by the global CNN.
- **Training Loss:** The loss reported from local client training for each federated round.
- **Attack Success Rate (ASR):** Percentage of triggered test images classified as the attacker’s configured target label.
- **Rejected Client ID:** The client update rejected by Krum in that round, compared with the known malicious client ID, `4`.

## Expected Output

The console prints clear progress logs for every round, including client roles, local training completion, Krum processing, the rejected client ID, clean accuracy, and ASR. A typical round summary is:

```text
====================================================
Round 1
====================================================

Client 0 : Honest
Training Completed

Client 1 : Honest
Training Completed

Client 2 : Honest
Training Completed

Client 3 : Honest
Training Completed

Client 4 : Malicious
Trigger Injection Performed

Receiving Client Updates...

Running Krum...

Rejected Client : 4

Aggregating Honest Updates...

Global Model Updated

Clean Accuracy : xx.xx %

Attack Success Rate : xx.xx %
====================================================
```

After training, the application saves only these graphs:

```text
graphs/accuracy_vs_round.png
graphs/asr_vs_round.png
```

## Future Improvements

- Compare Krum with FedAvg, Median, Trimmed Mean, and Multi-Krum.
- Support different trigger positions, shapes, and target classes.
- Experiment with IID and non-IID client data partitions.
- Evaluate different poison ratios and multiple malicious clients.
- Add cryptographic secure aggregation as a separate, advanced extension.
- Save experiment configurations and round metrics for repeated comparisons.
