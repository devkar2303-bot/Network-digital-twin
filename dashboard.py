"""
Network Digital Twin — Real-time Dashboard
Run: python dashboard.py
Simulates live telemetry and shows failure probability per link.
"""

import time
import random
import numpy as np
import os

try:
    from digital_twin import predict_live
    MODEL_READY = True
except Exception:
    MODEL_READY = False


def generate_live_sample(link_id, hour):
    """Simulate one live telemetry reading for a link."""
    util = np.clip(0.3 + 0.4 * np.sin(np.pi * hour / 12) + np.random.normal(0, 0.1), 0, 1)
    return {
        'utilisation':   round(util, 4),
        'latency_ms':    round(1 + util * 8 + abs(np.random.normal(0, 0.5)), 3),
        'packet_loss':   round(max(0, np.random.exponential(0.002)), 5),
        'error_rate':    round(max(0, np.random.exponential(0.001)), 5),
        'hour_of_day':   hour,
        'is_core_link':  random.randint(0, 1),
        'capacity_mbps': random.choice([100, 1000, 10000]),
        'util_lag1':     round(util + np.random.normal(0, 0.02), 4),
        'err_lag1':      0.001,
    }


def risk_bar(prob, width=20):
    filled = int(prob * width)
    bar = '█' * filled + '░' * (width - filled)
    if prob > 0.7:
        level = '🔴 HIGH'
    elif prob > 0.4:
        level = '🟡 MED '
    else:
        level = '🟢 LOW '
    return f"[{bar}] {prob:.0%} {level}"


def run_dashboard(n_links=6, refresh_sec=2, n_ticks=30):
    links = [f"node{i}-node{i+1}" for i in range(n_links)]
    hour  = 9  # start at 9 AM

    print("\n" + "="*60)
    print("  NETWORK DIGITAL TWIN — LIVE FAILURE RISK MONITOR")
    print("="*60)

    for tick in range(n_ticks):
        hour = (hour + 1) % 24
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"\n  Network Digital Twin  |  Tick {tick+1:02d}/{n_ticks}  |  Hour {hour:02d}:00")
        print("  " + "-"*56)

        for link in links:
            sample = generate_live_sample(link, hour)
            if MODEL_READY:
                prob = predict_live(sample)
            else:
                # fallback if model not trained yet
                prob = sample['utilisation'] * 0.4 + sample['error_rate'] * 3
                prob = min(prob, 1.0)

            bar = risk_bar(prob)
            print(f"  {link:16s}  util={sample['utilisation']:.2f}  {bar}")

        print("\n  Press Ctrl+C to stop")
        time.sleep(refresh_sec)

    print("\n  Simulation complete.")


if __name__ == '__main__':
    run_dashboard(n_links=6, refresh_sec=1.5, n_ticks=20)
