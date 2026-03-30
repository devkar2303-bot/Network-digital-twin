"""
Network Topology Simulator
Generates synthetic traffic and failure data for the Digital Twin.
"""

import numpy as np
import pandas as pd
import networkx as nx
import random
from datetime import datetime, timedelta


def build_topology(n_nodes=10, seed=42):
    """Create a random connected network graph simulating a small ISP."""
    random.seed(seed)
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(n_nodes, 2, seed=seed)
    for u, v in G.edges():
        G[u][v]['capacity_mbps'] = random.choice([100, 1000, 10000])
        G[u][v]['latency_ms']    = round(random.uniform(0.5, 20.0), 2)
    for node in G.nodes():
        G.nodes[node]['role'] = random.choice(['core', 'edge', 'access'])
    return G


def simulate_traffic(G, n_steps=2000, seed=0):
    """
    Simulate link-level telemetry over time.
    Each row = one link sample.  'failure' = 1 if the link was down.
    """
    np.random.seed(seed)
    records = []
    t0 = datetime(2024, 1, 1)

    for step in range(n_steps):
        ts = t0 + timedelta(minutes=step * 5)
        hour = ts.hour

        for u, v in G.edges():
            cap = G[u][v]['capacity_mbps']
            base_util = 0.3 + 0.4 * np.sin(np.pi * hour / 12)  # diurnal pattern
            util = np.clip(base_util + np.random.normal(0, 0.08), 0.0, 1.0)

            latency   = G[u][v]['latency_ms'] * (1 + util * 2 + np.random.normal(0, 0.05))
            pkt_loss  = max(0.0, np.random.exponential(0.002) + (0.03 if util > 0.9 else 0))
            err_rate  = max(0.0, np.random.exponential(0.001) + (0.05 if pkt_loss > 0.01 else 0))

            # Failure: high utilisation + high error rate → more likely
            fail_prob = 0.01 + 0.15 * (util > 0.92) + 0.20 * (err_rate > 0.04)
            failure   = int(np.random.random() < fail_prob)

            records.append({
                'timestamp':        ts.isoformat(),
                'link':             f'{u}-{v}',
                'node_u':           u,
                'node_v':           v,
                'capacity_mbps':    cap,
                'utilisation':      round(util, 4),
                'latency_ms':       round(latency, 3),
                'packet_loss':      round(pkt_loss, 5),
                'error_rate':       round(err_rate, 5),
                'hour_of_day':      hour,
                'is_core_link':     int(G.nodes[u]['role'] == 'core' or G.nodes[v]['role'] == 'core'),
                'failure':          failure,
            })

    df = pd.DataFrame(records)
    return df


if __name__ == '__main__':
    G  = build_topology(n_nodes=10)
    df = simulate_traffic(G, n_steps=2000)
    df.to_csv('data/synthetic_logs.csv', index=False)
    print(f"Generated {len(df):,} records  |  failure rate: {df['failure'].mean():.2%}")
    print(df.head(3).to_string())
