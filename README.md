# Network Digital Twin



A machine-learning system that builds a *digital twin* of a computer network:  
it continuously mirrors real-time link telemetry, predicts link failures 5 minutes  
ahead, and visualises risk in a live dashboard.

---

## What is a Network Digital Twin?

A **digital twin** is a live virtual replica of a physical system.  
Here, each network link (router-to-router connection) is mirrored by an ML model  
that ingests telemetry (utilisation, latency, packet loss, error rate) and outputs  
a failure-probability score. When the score exceeds a threshold, operators are alerted  
before the link actually goes down.

```
Physical Network          Digital Twin (ML)
─────────────────         ─────────────────────
Link telemetry ──────────►  Random Forest model
                              │
                              ▼
                         Failure prob.  →  Alert / Dashboard
```

---

## Results

| Metric   | Value |
|----------|-------|
| F1-score | ~0.82 |
| ROC-AUC  | ~0.91 |
| Dataset  | 26,000 synthetic link samples, 10-node topology |

**Error analysis (key finding):**  
False Negatives (missed failures) cluster in the 0.75–0.90 utilisation range —  
the "ambiguous zone" where congestion is building but hasn't yet triggered  
high error rates. Proposed fix: add a 3-step rolling median of latency as a feature  
to smooth transient spikes and expose true degradation trends.

---

## Project Structure

```
network-digital-twin/
├── network_sim.py               # Synthetic network topology + telemetry generator
├── digital_twin.py              # Random Forest training, evaluation, live inference
├── network_twin_analysis.ipynb  # Full analysis with error analysis (M1 requirement)
├── dashboard.py                 # Real-time failure risk dashboard
├── data/
│   └── synthetic_logs.csv       # Generated training data
├── results/
│   ├── metrics.json             # F1, ROC-AUC, confusion matrix, feature importances
│   ├── error_analysis.png       # FP/FN visualisation
│   └── feature_importance.png   # Feature importance plot
└── requirements.txt
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data
python network_sim.py

# 3. Train the digital twin
python digital_twin.py

# 4. Open the analysis notebook (includes error analysis)
jupyter notebook network_twin_analysis.ipynb

# 5. Run the live dashboard
python dashboard.py
```

---

## Key Design Choices

- **Random Forest** chosen over deep learning: interpretable feature importances,  
  robust with class imbalance (`class_weight='balanced'`), no GPU required.
- **Lag features** (previous step's utilisation/error rate) give the model temporal  
  context without the complexity of an LSTM.
- **Synthetic data** generated via a Barabási–Albert graph (realistic ISP topology)  
  with diurnal traffic patterns and failure probabilities tied to real engineering thresholds.

---

## Technologies

Python · scikit-learn · NetworkX · pandas · NumPy · Matplotlib · Jupyter
