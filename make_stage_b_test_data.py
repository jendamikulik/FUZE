import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# -------------------------
# Ringdown: alpha(t) = c + A exp(-t/tau)
# -------------------------
t_ring = np.linspace(0.0, 12.0, 220)
c = 0.12
A = 0.95
tau = 2.8
y_ring = c + A * np.exp(-t_ring / tau)
y_ring = y_ring + rng.normal(0.0, 0.015, size=t_ring.shape)

pd.DataFrame({
    "t": t_ring,
    "alpha": y_ring,
}).to_csv("ringdown.csv", index=False)

# -------------------------
# Closure: linear heat then exp cool
# -------------------------
t_clos = np.linspace(0.0, 20.0, 280)
T0 = 25.0
slope_heat = 1.8
t_switch = 7.5
tau_cool = 4.2

T_peak = T0 + slope_heat * (t_switch - t_clos.min())
y_clos = np.empty_like(t_clos)

heat = t_clos <= t_switch
cool = ~heat

y_clos[heat] = T0 + slope_heat * (t_clos[heat] - t_clos.min())
y_clos[cool] = T0 + (T_peak - T0) * np.exp(-(t_clos[cool] - t_switch) / tau_cool)
y_clos = y_clos + rng.normal(0.0, 0.25, size=t_clos.shape)

pd.DataFrame({
    "t": t_clos,
    "T": y_clos,
}).to_csv("closure.csv", index=False)

print("Saved ringdown.csv and closure.csv")