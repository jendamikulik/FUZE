import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pathlib import Path

outdir = Path("")

# ----------------------------
# FUZE — RTIM / A-B-C numerika
# ----------------------------

def lorentzian(omega, omega0, gamma):
    return gamma**2 / ((omega - omega0)**2 + gamma**2)

def build_params():
    return {
        "c_s": 0.80,
        "Gmax": 1.40,
        "alpha_sat": 0.75,
        "eta_in": 0.85,
        "eta_th": 0.90,
        "h": 0.16,
        "Cth": 9.5,
        "K_T": 1.0,
        "K_alpha": 0.25,
        "K_p": 0.80,
        "chi_alpha": 0.10,
        "tau0": 4.5,
        "tauT": 0.010,
        "gamma_loss0": 0.24,
        "gamma_lossT": 0.0015,
        "gamma_rel0": 0.38,
        "gamma_relT": 0.0010,
        "gamma_dump0": 0.18,
        "gamma_dumpT": 0.0010,
        "omega0_ref": 10.0,
        "domega_dT": -0.004,
        "gamma_ref": 0.90,
        "dgamma_dT": 0.001,
        "p_ohm_frac": 0.35,
        "T_amb": 25.0,
    }

def const_fun(v):
    return lambda t: v

def model_step(x, t, dt, p, controls, extra_channel=True):
    alpha, E_res, E_pair, T = x
    omega = controls["omega"](t)
    Pdrv = controls["Pdrv"](t)
    Q = controls["Q"](t)

    Tamb = p["T_amb"]
    omega0 = p["omega0_ref"] + p["domega_dT"] * (T - Tamb)
    gamma = max(0.05, p["gamma_ref"] + p["dgamma_dT"] * (T - Tamb))
    L = lorentzian(omega, omega0, gamma)

    tau_ent = max(0.05, p["tau0"] * np.exp(-p["tauT"] * max(T - Tamb, 0.0)))
    gamma_loss = max(0.0, p["gamma_loss0"] + p["gamma_lossT"] * max(T - Tamb, 0.0))
    gamma_rel = max(0.0, p["gamma_rel0"] + p["gamma_relT"] * max(T - Tamb, 0.0))
    gamma_dump = max(0.0, p["gamma_dump0"] + p["gamma_dumpT"] * max(T - Tamb, 0.0))

    alpha_drive = p["c_s"] * Pdrv * Q * L
    gamma_tr = p["Gmax"] * alpha / (alpha + p["alpha_sat"] + 1e-12)

    dalpha = alpha_drive - alpha / tau_ent - p["chi_alpha"] * alpha
    dE_res = p["eta_in"] * Pdrv - (gamma_loss + gamma_tr) * E_res

    if extra_channel:
        dE_pair = gamma_tr * E_res - (gamma_rel + gamma_dump) * E_pair
        thermal_extra = p["eta_th"] * gamma_dump * E_pair
    else:
        dE_pair = -0.5 * E_pair
        thermal_extra = 0.0

    dT = (p["p_ohm_frac"] * Pdrv + thermal_extra - p["h"] * (T - Tamb)) / p["Cth"]

    alpha = max(0.0, alpha + dt * dalpha)
    E_res = max(0.0, E_res + dt * dE_res)
    E_pair = max(0.0, E_pair + dt * dE_pair)
    T = T + dt * dT

    z = p["K_T"] * T + p["K_alpha"] * alpha + p["K_p"] * E_pair
    aux = {
        "omega": omega,
        "omega0": omega0,
        "gamma": gamma,
        "L": L,
        "gamma_tr": gamma_tr,
        "tau_ent": tau_ent,
    }
    return np.array([alpha, E_res, E_pair, T]), z, aux

def simulate(times, p, controls, x0=None, extra_channel=True):
    x = np.array([0.0, 0.0, 0.0, p["T_amb"]] if x0 is None else x0, dtype=float)
    xs = np.zeros((len(times), 4))
    zs = np.zeros(len(times))
    auxs = []
    xs[0] = x
    zs[0] = p["K_T"] * x[3] + p["K_alpha"] * x[0] + p["K_p"] * x[2]
    auxs.append({})
    for k in range(1, len(times)):
        x, z, aux = model_step(x, times[k - 1], times[k] - times[k - 1], p, controls, extra_channel=extra_channel)
        xs[k] = x
        zs[k] = z
        auxs.append(aux)
    return xs, zs, auxs

def steady_signal(omega, Pdrv, p, extra_channel=False):
    t = np.linspace(0, 30, 301)
    controls = {"omega": const_fun(omega), "Pdrv": const_fun(Pdrv), "Q": const_fun(20.0)}
    xs, zs, auxs = simulate(t, p, controls, extra_channel=extra_channel)
    return zs[-50:].mean(), xs[-1], auxs[-1]

# True synthetic system
true_params = build_params()
rng = np.random.default_rng(0)

# A) benigní resonance sweep
omegas = np.linspace(7.0, 13.0, 15)
drive_levels = [4.0, 6.0]
A_data = []
for Pdrv in drive_levels:
    y = np.array([steady_signal(w, Pdrv, true_params, extra_channel=False)[0] for w in omegas])
    y = y + rng.normal(0, 0.02, len(y))
    A_data.append((Pdrv, y))

# A) ringdown
times_pre = np.linspace(0, 30, 301)
controls_on = {"omega": const_fun(10.0), "Pdrv": const_fun(6.0), "Q": const_fun(20.0)}
x_pre, _, _ = simulate(times_pre, true_params, controls_on, extra_channel=False)
x0_ring = x_pre[-1]
times_ring = np.linspace(0, 10, 201)
controls_off = {"omega": const_fun(10.0), "Pdrv": const_fun(0.0), "Q": const_fun(20.0)}
ring_xs, _, _ = simulate(times_ring, true_params, controls_off, x0=x0_ring, extra_channel=False)
ring_alpha = ring_xs[:, 0] + rng.normal(0, 0.01, len(times_ring))

# B/C pulses
def pulse(t):
    return 0.0 if t < 5 else (7.0 if t < 40 else 0.0)

times_pulse = np.linspace(0, 60, 601)
controls_pulse = {"omega": const_fun(10.0), "Pdrv": pulse, "Q": const_fun(20.0)}

B_xs, _, _ = simulate(times_pulse, true_params, controls_pulse, extra_channel=False)
T_B_meas = B_xs[:, 3] + rng.normal(0, 0.03, len(times_pulse))

C_xs, C_zs, _ = simulate(times_pulse, true_params, controls_pulse, extra_channel=True)
T_C_meas = C_xs[:, 3] + rng.normal(0, 0.03, len(times_pulse))
z_C_meas = C_zs + rng.normal(0, 0.04, len(times_pulse))

# RTIM fit — intentionally only a subset to keep it identifiable/light
fit_names = ["c_s", "gamma_ref", "tau0", "K_p"]
p0 = np.array([0.60, 1.20, 3.0, 0.50], dtype=float)
lb = np.array([0.1, 0.2, 0.5, 0.0])
ub = np.array([2.0, 2.5, 12.0, 4.0])

def pack(v):
    p = build_params()
    for k, val in zip(fit_names, v):
        p[k] = float(val)
    return p

def residuals(v):
    p = pack(v)
    res = []

    # A: sweeps
    for Pdrv, meas in A_data:
        pred = np.array([steady_signal(w, Pdrv, p, extra_channel=False)[0] for w in omegas])
        res.extend((pred - meas) / 0.03)

    # A: ringdown
    x_pre_local, _, _ = simulate(times_pre, p, controls_on, extra_channel=False)
    ring_fit, _, _ = simulate(times_ring, p, controls_off, x0=x_pre_local[-1], extra_channel=False)
    res.extend((ring_fit[:, 0] - ring_alpha) / 0.02)

    # B: thermal closure
    B_fit_xs, _, _ = simulate(times_pulse, p, controls_pulse, extra_channel=False)
    res.extend((B_fit_xs[:, 3] - T_B_meas) / 0.05)

    # C: extra channel
    C_fit_xs, C_fit_zs, _ = simulate(times_pulse, p, controls_pulse, extra_channel=True)
    res.extend((C_fit_xs[:, 3] - T_C_meas) / 0.05)
    res.extend((C_fit_zs - z_C_meas) / 0.06)

    return np.array(res)

fit = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=20, verbose=0)
fit_params = pack(fit.x)

# Predictions
A_fit = []
for Pdrv, _ in A_data:
    pred = np.array([steady_signal(w, Pdrv, fit_params, extra_channel=False)[0] for w in omegas])
    A_fit.append((Pdrv, pred))

x_pre_fit, _, _ = simulate(times_pre, fit_params, controls_on, extra_channel=False)
ring_fit, _, _ = simulate(times_ring, fit_params, controls_off, x0=x_pre_fit[-1], extra_channel=False)
B_fit_xs, _, _ = simulate(times_pulse, fit_params, controls_pulse, extra_channel=False)
C_fit_xs, C_fit_zs, _ = simulate(times_pulse, fit_params, controls_pulse, extra_channel=True)

# Plot
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.22)

ax1 = fig.add_subplot(gs[0, 0])
for (Pdrv, meas), (_, pred) in zip(A_data, A_fit):
    ax1.plot(omegas, meas, "o", ms=4, alpha=0.75, label=f"data P={Pdrv}")
    ax1.plot(omegas, pred, "-", lw=2, label=f"fit P={Pdrv}")
ax1.set_title("A) Benigní resonance sweep")
ax1.set_xlabel("ω")
ax1.set_ylabel("z_ss")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(times_ring, ring_alpha, ".", ms=2, alpha=0.5, label="data α(t)")
ax2.plot(times_ring, ring_fit[:, 0], "-", lw=2, label="fit α(t)")
ax2.set_title("A) Ringdown")
ax2.set_xlabel("t")
ax2.set_ylabel("α(t)")
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(times_pulse, T_B_meas, ".", ms=2, alpha=0.5, label="B data T")
ax3.plot(times_pulse, B_fit_xs[:, 3], "-", lw=2, label="B fit T")
ax3.set_title("B) Kalorimetrická uzávěra")
ax3.set_xlabel("t")
ax3.set_ylabel("T(t)")
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(times_pulse, T_C_meas, ".", ms=2, alpha=0.35, label="C data T")
ax4.plot(times_pulse, C_fit_xs[:, 3], "-", lw=2, label="C fit T")
ax4.plot(times_pulse, z_C_meas, ".", ms=2, alpha=0.25, label="C data z")
ax4.plot(times_pulse, C_fit_zs, "-", lw=2, label="C fit z")
ax4.set_title("C) Hypotetický extra channel")
ax4.set_xlabel("t")
ax4.legend(fontsize=8)

ax5 = fig.add_subplot(gs[2, :])
labels = fit_names
true_vals = [true_params[k] for k in fit_names]
est_vals = [fit_params[k] for k in fit_names]
x = np.arange(len(labels))
w = 0.35
ax5.bar(x - w/2, true_vals, width=w, label="true")
ax5.bar(x + w/2, est_vals, width=w, label="fit")
ax5.set_xticks(x)
ax5.set_xticklabels(labels)
ax5.set_title("FUZE / RTIM — identifikovaná podmnožina parametrů")
ax5.legend()

report_path = outdir / "fuze_report.png"
fig.savefig(report_path, dpi=180, bbox_inches="tight")
plt.close(fig)

# Save full script
script_text = f"""import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# FUZE — Resonant Transfer Identification Model (RTIM)

{Path('/proc/self/cmdline').read_text(errors='ignore') if False else ''}"""
# Better: write the current notebook code in a simpler maintained script body.
script_body = r'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def lorentzian(omega, omega0, gamma):
    return gamma**2 / ((omega - omega0)**2 + gamma**2)

def build_params():
    return {
        "c_s": 0.80, "Gmax": 1.40, "alpha_sat": 0.75, "eta_in": 0.85, "eta_th": 0.90,
        "h": 0.16, "Cth": 9.5, "K_T": 1.0, "K_alpha": 0.25, "K_p": 0.80,
        "chi_alpha": 0.10, "tau0": 4.5, "tauT": 0.010,
        "gamma_loss0": 0.24, "gamma_lossT": 0.0015,
        "gamma_rel0": 0.38, "gamma_relT": 0.0010,
        "gamma_dump0": 0.18, "gamma_dumpT": 0.0010,
        "omega0_ref": 10.0, "domega_dT": -0.004,
        "gamma_ref": 0.90, "dgamma_dT": 0.001,
        "p_ohm_frac": 0.35, "T_amb": 25.0,
    }

def const_fun(v):
    return lambda t: v

def model_step(x, t, dt, p, controls, extra_channel=True):
    alpha, E_res, E_pair, T = x
    omega = controls["omega"](t)
    Pdrv = controls["Pdrv"](t)
    Q = controls["Q"](t)

    Tamb = p["T_amb"]
    omega0 = p["omega0_ref"] + p["domega_dT"] * (T - Tamb)
    gamma = max(0.05, p["gamma_ref"] + p["dgamma_dT"] * (T - Tamb))
    L = lorentzian(omega, omega0, gamma)

    tau_ent = max(0.05, p["tau0"] * np.exp(-p["tauT"] * max(T - Tamb, 0.0)))
    gamma_loss = max(0.0, p["gamma_loss0"] + p["gamma_lossT"] * max(T - Tamb, 0.0))
    gamma_rel = max(0.0, p["gamma_rel0"] + p["gamma_relT"] * max(T - Tamb, 0.0))
    gamma_dump = max(0.0, p["gamma_dump0"] + p["gamma_dumpT"] * max(T - Tamb, 0.0))

    alpha_drive = p["c_s"] * Pdrv * Q * L
    gamma_tr = p["Gmax"] * alpha / (alpha + p["alpha_sat"] + 1e-12)

    dalpha = alpha_drive - alpha / tau_ent - p["chi_alpha"] * alpha
    dE_res = p["eta_in"] * Pdrv - (gamma_loss + gamma_tr) * E_res

    if extra_channel:
        dE_pair = gamma_tr * E_res - (gamma_rel + gamma_dump) * E_pair
        thermal_extra = p["eta_th"] * gamma_dump * E_pair
    else:
        dE_pair = -0.5 * E_pair
        thermal_extra = 0.0

    dT = (p["p_ohm_frac"] * Pdrv + thermal_extra - p["h"] * (T - Tamb)) / p["Cth"]

    alpha = max(0.0, alpha + dt * dalpha)
    E_res = max(0.0, E_res + dt * dE_res)
    E_pair = max(0.0, E_pair + dt * dE_pair)
    T = T + dt * dT

    z = p["K_T"] * T + p["K_alpha"] * alpha + p["K_p"] * E_pair
    aux = {"omega": omega, "omega0": omega0, "gamma": gamma, "L": L, "gamma_tr": gamma_tr}
    return np.array([alpha, E_res, E_pair, T]), z, aux

def simulate(times, p, controls, x0=None, extra_channel=True):
    x = np.array([0.0, 0.0, 0.0, p["T_amb"]] if x0 is None else x0, dtype=float)
    xs = np.zeros((len(times), 4))
    zs = np.zeros(len(times))
    auxs = []
    xs[0] = x
    zs[0] = p["K_T"] * x[3] + p["K_alpha"] * x[0] + p["K_p"] * x[2]
    auxs.append({})
    for k in range(1, len(times)):
        x, z, aux = model_step(x, times[k - 1], times[k] - times[k - 1], p, controls, extra_channel=extra_channel)
        xs[k] = x
        zs[k] = z
        auxs.append(aux)
    return xs, zs, auxs

def steady_signal(omega, Pdrv, p, extra_channel=False):
    t = np.linspace(0, 30, 301)
    controls = {"omega": const_fun(omega), "Pdrv": const_fun(Pdrv), "Q": const_fun(20.0)}
    xs, zs, auxs = simulate(t, p, controls, extra_channel=extra_channel)
    return zs[-50:].mean(), xs[-1], auxs[-1]

def run_demo():
    true_params = build_params()
    rng = np.random.default_rng(0)

    omegas = np.linspace(7.0, 13.0, 15)
    drive_levels = [4.0, 6.0]
    A_data = []
    for Pdrv in drive_levels:
        y = np.array([steady_signal(w, Pdrv, true_params, extra_channel=False)[0] for w in omegas])
        y = y + rng.normal(0, 0.02, len(y))
        A_data.append((Pdrv, y))

    times_pre = np.linspace(0, 30, 301)
    controls_on = {"omega": const_fun(10.0), "Pdrv": const_fun(6.0), "Q": const_fun(20.0)}
    x_pre, _, _ = simulate(times_pre, true_params, controls_on, extra_channel=False)
    x0_ring = x_pre[-1]
    times_ring = np.linspace(0, 10, 201)
    controls_off = {"omega": const_fun(10.0), "Pdrv": const_fun(0.0), "Q": const_fun(20.0)}
    ring_xs, _, _ = simulate(times_ring, true_params, controls_off, x0=x0_ring, extra_channel=False)
    ring_alpha = ring_xs[:, 0] + rng.normal(0, 0.01, len(times_ring))

    def pulse(t):
        return 0.0 if t < 5 else (7.0 if t < 40 else 0.0)

    times_pulse = np.linspace(0, 60, 601)
    controls_pulse = {"omega": const_fun(10.0), "Pdrv": pulse, "Q": const_fun(20.0)}

    B_xs, _, _ = simulate(times_pulse, true_params, controls_pulse, extra_channel=False)
    T_B_meas = B_xs[:, 3] + rng.normal(0, 0.03, len(times_pulse))

    C_xs, C_zs, _ = simulate(times_pulse, true_params, controls_pulse, extra_channel=True)
    T_C_meas = C_xs[:, 3] + rng.normal(0, 0.03, len(times_pulse))
    z_C_meas = C_zs + rng.normal(0, 0.04, len(times_pulse))

    fit_names = ["c_s", "gamma_ref", "tau0", "K_p"]
    p0 = np.array([0.60, 1.20, 3.0, 0.50], dtype=float)
    lb = np.array([0.1, 0.2, 0.5, 0.0])
    ub = np.array([2.0, 2.5, 12.0, 4.0])

    def pack(v):
        p = build_params()
        for k, val in zip(fit_names, v):
            p[k] = float(val)
        return p

    def residuals(v):
        p = pack(v)
        res = []
        for Pdrv, meas in A_data:
            pred = np.array([steady_signal(w, Pdrv, p, extra_channel=False)[0] for w in omegas])
            res.extend((pred - meas) / 0.03)
        x_pre_local, _, _ = simulate(times_pre, p, controls_on, extra_channel=False)
        ring_fit, _, _ = simulate(times_ring, p, controls_off, x0=x_pre_local[-1], extra_channel=False)
        res.extend((ring_fit[:, 0] - ring_alpha) / 0.02)
        B_fit_xs, _, _ = simulate(times_pulse, p, controls_pulse, extra_channel=False)
        res.extend((B_fit_xs[:, 3] - T_B_meas) / 0.05)
        C_fit_xs, C_fit_zs, _ = simulate(times_pulse, p, controls_pulse, extra_channel=True)
        res.extend((C_fit_xs[:, 3] - T_C_meas) / 0.05)
        res.extend((C_fit_zs - z_C_meas) / 0.06)
        return np.array(res)

    fit = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=20, verbose=1)
    fit_params = pack(fit.x)
    print("Fit success:", fit.success)
    for name in fit_names:
        print(name, "true =", true_params[name], "fit =", fit_params[name])

if __name__ == "__main__":
    run_demo()
'''
script_path = outdir / "fuze_rtim.py"
script_path.write_text(script_body, encoding="utf-8")

# Save a tiny markdown note
note = f"""# FUZE / RTIM

Tahle numerika je rozdělená do tří vrstev:

- **A** — benigní resonance sweep + ringdown
- **B** — kalorimetrická uzávěra bez extra channel
- **C** — hypotetický extra channel

Stavový model:
\\[
x(t)=(\\alpha(t),E_{{res}}(t),E_{{pair}}(t),T(t))
\\]

Readout:
\\[
z(t)=K_T T(t)+K_\\alpha \\alpha(t)+K_p E_{{pair}}(t).
\\]

Identifikace:
fituje se malá podmnožina parametrů
\\[
(c_s,\\gamma_{{ref}},\\tau_0,K_p)
\\]
přes všechny tři vrstvy najednou.

## Výsledek demoverze
- fit success: **{fit.success}**
- cost: **{fit.cost:.3f}**
- resonance center (true): **{true_params['omega0_ref']:.3f}**
- fitted gamma_ref: **{fit_params['gamma_ref']:.4f}**
- true/fitted peak \\(E_{{pair}}\\): **{C_xs[:,2].max():.3f} / {C_fit_xs[:,2].max():.3f}**

Tohle je identifikační skeleton, ne důkaz realizované LENR.
"""
note_path = outdir / "fuze_note.md"
note_path.write_text(note, encoding="utf-8")

print(f"Fit success: {fit.success}")
print(f"Cost: {fit.cost:.3f}")
for name in fit_names:
    print(f"{name}: true={true_params[name]:.4f}, fit={fit_params[name]:.4f}")
print(f"Saved: {script_path.name}, {report_path.name}, {note_path.name}")