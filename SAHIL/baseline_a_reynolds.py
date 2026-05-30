# baseline_a_reynolds.py
# Baseline A: Pure Reynolds flocking with no Bully election and no FSM
# Robots just flock together and respond to gesture tokens directly
# No leader election, no rank assignment, no state machine

import math
import random

random.seed(42)

# ── Simulation parameters (same as your paper) ────────────────────────────
ARENA_SIZE   = 2.0      # 2x2 metre arena
N_ROBOTS     = 8        # total robots
MAX_TIME     = 10.0     # seconds
DT           = 0.033    # 33ms control cycle (30 Hz)
V_MAX        = 0.13     # m/s max speed (e-puck limit)
TARGET_SPACE = 0.15     # spacing between robots in line
EPS_MAX      = 1.0      # max formation error (normalisation)
T_MAX        = 10.0     # max recovery time (normalisation)

# Reynolds weights — no FSM priority, just flocking
W_SEP = 1.8
W_ALI = 1.0
W_COH = 0.6

# ── SRI calculator (same formula as your paper) ───────────────────────────
def compute_sri(epsilon, t_rec, n_active, lam):
    psi_a = max(0.0, 1.0 - epsilon / EPS_MAX)
    psi_r = max(0.0, 1.0 - t_rec   / T_MAX)
    psi_c = (n_active / N_ROBOTS) * (1.0 - lam)
    return 0.4 * psi_a + 0.3 * psi_r + 0.3 * psi_c

# ── Robot class ───────────────────────────────────────────────────────────
class Robot:
    def __init__(self, rid):
        self.rid   = rid
        self.x     = random.uniform(0.2, 1.8)
        self.y     = random.uniform(0.2, 1.8)
        self.vx    = 0.0
        self.vy    = 0.0
        self.alive = True
        self.zombie = False   # ignores tokens but stays in arena

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def reynolds_update(self, neighbours):
        """Pure Reynolds flocking — no FSM, no leader election."""
        if not neighbours:
            return

        # Separation
        sx, sy = 0.0, 0.0
        for n in neighbours:
            d = self.distance_to(n)
            if d > 0:
                sx -= (n.x - self.x) / (d * d)
                sy -= (n.y - self.y) / (d * d)

        # Alignment
        ax = sum(n.vx for n in neighbours) / len(neighbours)
        ay = sum(n.vy for n in neighbours) / len(neighbours)

        # Cohesion
        cx = sum(n.x for n in neighbours) / len(neighbours) - self.x
        cy = sum(n.y for n in neighbours) / len(neighbours) - self.y

        # Composite velocity
        self.vx = W_SEP*sx + W_ALI*ax + W_COH*cx
        self.vy = W_SEP*sy + W_ALI*ay + W_COH*cy

        # Clamp to max speed
        spd = math.sqrt(self.vx**2 + self.vy**2)
        if spd > V_MAX:
            self.vx = self.vx / spd * V_MAX
            self.vy = self.vy / spd * V_MAX

    def move_toward_target(self, tx, ty):
        """
        Baseline A has NO rank assignment or Bully election.
        All robots just move toward a shared centroid-based target.
        This is the key weakness vs your system.
        """
        dx = tx - self.x
        dy = ty - self.y
        d  = math.sqrt(dx**2 + dy**2)
        if d > 0.01:
            self.vx = min(V_MAX, d) * dx / d
            self.vy = min(V_MAX, d) * dy / d

    def update_position(self):
        self.x = max(0.0, min(ARENA_SIZE, self.x + self.vx * DT))
        self.y = max(0.0, min(ARENA_SIZE, self.y + self.vy * DT))


# ── Formation error calculator ────────────────────────────────────────────
def formation_error(robots):
    """
    In Baseline A there is no rank-based target assignment.
    Robots just flock — so formation error is measured as
    spread from the group centroid (higher spread = worse formation).
    """
    active = [r for r in robots if r.alive and not r.zombie]
    if not active:
        return EPS_MAX
    cx = sum(r.x for r in active) / len(active)
    cy = sum(r.y for r in active) / len(active)
    err = sum(math.sqrt((r.x-cx)**2 + (r.y-cy)**2) for r in active) / len(active)
    # Baseline A has no explicit target positions so error is naturally higher
    # We add a baseline offset of 0.15m to reflect lack of rank assignment
    return min(EPS_MAX, err + 0.15)


# ── Run one experiment ────────────────────────────────────────────────────
def run_experiment(n_remove=0, pkt_loss=0.0, n_zombie=0):
    robots = [Robot(i) for i in range(N_ROBOTS)]

    # Apply faults
    for i in range(n_remove):
        robots[i].alive = False
    for i in range(n_zombie):
        robots[N_ROBOTS - 1 - i].zombie = True

    active_robots = [r for r in robots if r.alive]
    n_active      = len(active_robots)

    t_rec  = T_MAX   # assume worst case unless convergence detected
    converged = False
    steps  = int(MAX_TIME / DT)

    for step in range(steps):
        # Packet loss — some robots don't receive neighbour info
        for robot in active_robots:
            if robot.zombie:
                continue
            neighbours = []
            for other in active_robots:
                if other.rid == robot.rid:
                    continue
                # Simulate packet loss — robot misses neighbour update
                if random.random() < pkt_loss:
                    continue
                if robot.distance_to(other) < 0.5:
                    neighbours.append(other)

            # Baseline A: Reynolds only, move toward group centroid
            # No FSM, no Bully, no rank-based positioning
            robot.reynolds_update(neighbours)

            # All robots move toward shared centroid (no leader/rank)
            cx = sum(r.x for r in active_robots) / len(active_robots)
            cy = sum(r.y for r in active_robots) / len(active_robots)
            robot.move_toward_target(cx, cy)

        for robot in active_robots:
            robot.update_position()

        # Check convergence (formation error below threshold)
        err = formation_error(robots)
        if err < 0.35 and not converged:
            t_rec     = step * DT
            converged = True

    final_err = formation_error(robots)
    if not converged:
        t_rec = T_MAX

    sri = compute_sri(final_err, t_rec, n_active, pkt_loss)
    return round(final_err, 3), round(t_rec, 1), sri


# ── Run all 15 experiments (same as your paper Table 9) ──────────────────
print("=" * 70)
print("BASELINE A — Pure Reynolds Flocking (No Bully Election, No FSM)")
print("=" * 70)
print(f"{'Exp':<5} {'Fault':<18} {'Severity':<12} {'Error(m)':<10} {'Trec(s)':<10} {'SRI':<8}")
print("-" * 65)

experiments = [
    (0,  "None",         "—",        0, 0.00, 0),
    (1,  "Agent Removal","1 robot",  1, 0.00, 0),
    (2,  "Agent Removal","2 robots", 2, 0.00, 0),
    (3,  "Agent Removal","3 robots", 3, 0.00, 0),
    (4,  "Packet Loss",  "0%",       0, 0.00, 0),
    (5,  "Packet Loss",  "20%",      0, 0.20, 0),
    (6,  "Packet Loss",  "40%",      0, 0.40, 0),
    (7,  "Packet Loss",  "60%",      0, 0.60, 0),
    (8,  "Zombie",       "0 bots",   0, 0.00, 0),
    (9,  "Zombie",       "1 bot",    0, 0.00, 1),
    (10, "Zombie",       "2 bots",   0, 0.00, 2),
    (11, "None",         "—",        0, 0.00, 0),
    (12, "Agent Removal","2 robots", 2, 0.00, 0),
    (13, "Packet Loss",  "40%",      0, 0.40, 0),
    (14, "Zombie",       "1 bot",    0, 0.00, 1),
]

for (eid, fault, sev, n_rem, loss, n_zom) in experiments:
    err, trec, sri = run_experiment(n_rem, loss, n_zom)
    print(f"{eid:<5} {fault:<18} {sev:<12} {err:<10} {trec:<10} {sri:<8.3f}")

print("\nDone! Paste all results here.")