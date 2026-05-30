# sensitivity_analysis.py
# Run this with: python sensitivity_analysis.py

import numpy as np

# ── Your 15 experiment results from Table 9 of the paper ──────────────────
experiments = [
    {"id": 0,  "fault": "None (baseline)",  "sev": "—",       "n_active": 8, "eps": 0.359, "Trec": 9.9, "lam": 0.00},
    {"id": 1,  "fault": "Agent Removal",    "sev": "1 robot",  "n_active": 7, "eps": 0.374, "Trec": 9.9, "lam": 0.00},
    {"id": 2,  "fault": "Agent Removal",    "sev": "2 robots", "n_active": 6, "eps": 0.357, "Trec": 9.9, "lam": 0.00},
    {"id": 3,  "fault": "Agent Removal",    "sev": "3 robots", "n_active": 5, "eps": 0.346, "Trec": 9.9, "lam": 0.00},
    {"id": 4,  "fault": "Packet Loss",      "sev": "0%",       "n_active": 8, "eps": 0.388, "Trec": 9.9, "lam": 0.00},
    {"id": 5,  "fault": "Packet Loss",      "sev": "20%",      "n_active": 8, "eps": 0.387, "Trec": 9.9, "lam": 0.20},
    {"id": 6,  "fault": "Packet Loss",      "sev": "40%",      "n_active": 8, "eps": 0.384, "Trec": 9.9, "lam": 0.40},
    {"id": 7,  "fault": "Packet Loss",      "sev": "60%",      "n_active": 8, "eps": 0.393, "Trec": 9.9, "lam": 0.60},
    {"id": 8,  "fault": "Zombie Agent",     "sev": "0 bots",   "n_active": 8, "eps": 0.393, "Trec": 9.9, "lam": 0.00},
    {"id": 9,  "fault": "Zombie Agent",     "sev": "1 bot",    "n_active": 8, "eps": 0.426, "Trec": 9.9, "lam": 0.00},
    {"id": 10, "fault": "Zombie Agent",     "sev": "2 bots",   "n_active": 8, "eps": 0.464, "Trec": 9.9, "lam": 0.00},
    {"id": 11, "fault": "None (baseline)",  "sev": "—",       "n_active": 8, "eps": 0.367, "Trec": 9.9, "lam": 0.00},
    {"id": 12, "fault": "Agent Removal",    "sev": "2 robots", "n_active": 6, "eps": 0.362, "Trec": 9.9, "lam": 0.00},
    {"id": 13, "fault": "Packet Loss",      "sev": "40%",      "n_active": 8, "eps": 0.358, "Trec": 9.9, "lam": 0.40},
    {"id": 14, "fault": "Zombie Agent",     "sev": "1 bot",    "n_active": 8, "eps": 0.358, "Trec": 9.9, "lam": 0.00},
]

# ── Calculate the 3 sub-scores for each experiment ────────────────────────
for e in experiments:
    e["psi_a"] = max(0.0, 1.0 - e["eps"] / 1.0)
    e["psi_r"] = max(0.0, 1.0 - e["Trec"] / 10.0)
    e["psi_c"] = (e["n_active"] / 8.0) * (1.0 - e["lam"])

# ── PART 1: Show SRI for 3 different weight choices ───────────────────────
print("=" * 65)
print("PART 1: SRI values under 3 different weight choices")
print("=" * 65)
print(f"{'ID':<4} {'Fault':<18} {'Sev':<10} {'(0.4,0.3,0.3)':>14} {'(0.5,0.3,0.2)':>14} {'(0.3,0.4,0.3)':>14}")
print("-" * 75)
for e in experiments:
    s1 = 0.4*e["psi_a"] + 0.3*e["psi_r"] + 0.3*e["psi_c"]
    s2 = 0.5*e["psi_a"] + 0.3*e["psi_r"] + 0.2*e["psi_c"]
    s3 = 0.3*e["psi_a"] + 0.4*e["psi_r"] + 0.3*e["psi_c"]
    print(f"{e['id']:<4} {e['fault']:<18} {e['sev']:<10} {s1:>14.3f} {s2:>14.3f} {s3:>14.3f}")

# ── PART 2: Test 69 weight combos (restricted range 0.15 to 0.60) ─────────
print("\n" + "=" * 65)
print("PART 2: Monotonicity check across 69 weight combinations")
print("(All weights kept between 0.15 and 0.60 — the practical range)")
print("=" * 65)

restricted_weights = []
for w1 in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    for w2 in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        w3 = round(1.0 - w1 - w2, 2)
        if 0.15 <= w3 <= 0.60:
            restricted_weights.append((w1, w2, w3))

print(f"Total weight combinations tested: {len(restricted_weights)}")
print()

categories = {
    "Agent Removal  (Exp 0 > 1 > 2 > 3)": [0, 1, 2, 3],
    "Packet Loss    (Exp 4 > 5 > 6 > 7)": [4, 5, 6, 7],
    "Zombie Agents  (Exp 8 > 9 > 10)   ": [8, 9, 10],
}

for cat_name, ids in categories.items():
    failures = 0
    for w1, w2, w3 in restricted_weights:
        sris = [w1*experiments[i]["psi_a"] + w2*experiments[i]["psi_r"] + w3*experiments[i]["psi_c"] for i in ids]
        is_monotone = all(sris[j] >= sris[j+1] for j in range(len(sris)-1))
        if not is_monotone:
            failures += 1
    pct = 100 * (len(restricted_weights) - failures) / len(restricted_weights)
    print(f"{cat_name}  ->  {len(restricted_weights)-failures}/{len(restricted_weights)} monotone ({pct:.1f}%)")

# ── PART 3: Simple monotonicity check with original weights ───────────────
print("\n" + "=" * 65)
print("PART 3: Does more fault = lower SRI? (original weights 0.4,0.3,0.3)")
print("=" * 65)
def sri(e, w1=0.4, w2=0.3, w3=0.3):
    return w1*e["psi_a"] + w2*e["psi_r"] + w3*e["psi_c"]

E = experiments
print(f"Agent Removal:  Exp0={sri(E[0]):.3f} > Exp1={sri(E[1]):.3f} > Exp2={sri(E[2]):.3f} > Exp3={sri(E[3]):.3f}  ->  {'YES' if sri(E[0])>sri(E[1])>sri(E[2])>sri(E[3]) else 'NO'}")
print(f"Packet Loss:    Exp4={sri(E[4]):.3f} > Exp5={sri(E[5]):.3f} > Exp6={sri(E[6]):.3f} > Exp7={sri(E[7]):.3f}  ->  {'YES' if sri(E[4])>sri(E[5])>sri(E[6])>sri(E[7]) else 'NO'}")
print(f"Zombie Agents:  Exp8={sri(E[8]):.3f} > Exp9={sri(E[9]):.3f} > Exp10={sri(E[10]):.3f}                       ->  {'YES' if sri(E[8])>sri(E[9])>sri(E[10]) else 'NO'}")

print("\nDone! Copy and paste all of this output back to Claude.")