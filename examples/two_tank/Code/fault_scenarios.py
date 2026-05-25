"""
fault_scenarios.py
══════════════════
Störfall-Szenarien für Kapitel 4.2 der Bachelorarbeit.

Drei Szenarien werden simuliert und visualisiert:
  1. Ventilausfall   – Zwischenventil schließt sich bei t = fault_time
  2. Pumpenausfall   – Pumpe eines Tanks fällt bei t = fault_time aus
  3. Leckage         – Leckagekoeffizient Qf steigt bei t = fault_time stark an

Jedes Szenario vergleicht Normal- vs. Störfall-Verlauf im selben Plot.

Verwendung (standalone):
    python fault_scenarios.py               # lädt config.yaml
    python fault_scenarios.py my_config.yaml

Verwendung (aus experiment_runner.py):
    from fault_scenarios import run_fault_scenarios
    run_fault_scenarios(n_tanks, topology, qp_list, h_target, areas)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl
import yaml

import flowcean.cli
from flowcean.ode import OdeEnvironment, OdeState, OdeSystem
from flowcean.polars import collect
from flowcean.utils.random import initialize_random
from numpy.typing import NDArray
from typing_extensions import Self, override

# ── Importiere Basisklassen aus Hauptdatei ────────────────────────────────────
from ntank_simulation import (
    FrameCollector,
    NTankState,
    Valve,
)

OUTPUT_DIR = Path.cwd() / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FAULT-DATENSTRUKTUR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Fault:
    """
    Beschreibt einen Störfall im N-Tank-System.

    Attribute:
        fault_type   – Art des Störfalls:
                       "valve_stuck"    → Zwischenventil klemmt (schließt sich)
                       "pump_failure"   → Pumpe fällt aus (Qpmax → 0)
                       "pump_rate_change" → Pumpenrate ändert sich (Qpmax → magnitude)
                       "leak_increase"  → Leckage erhöht sich stark
        start_time   – Zeitpunkt t [s] ab dem der Störfall eintritt
        tank_idx     – betroffener Tank (0-basiert); bei valve_stuck: Ventil-Index
        magnitude    – Stärke des Störfalls:
                       valve_stuck:   neue Position (0.0 = vollständig zu)
                       pump_failure:  nicht verwendet (Pumpe = 0)
                       pump_rate_change: neue Pumpenrate Qpmax (z.B. 0.5 = halbe Rate)
                       leak_increase: neuer Qf-Multiplikator (z.B. 10.0 = 10× mehr Leck)
    """
    fault_type:  str
    start_time:  float
    tank_idx:    int   = 0
    magnitude:   float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  FAULT-FÄHIGE ODE-SYSTEME
# ══════════════════════════════════════════════════════════════════════════════

class NTankLinearFault(OdeSystem):
    """
    Erweiterte lineare Tanktopologie mit Störfall-Unterstützung.
    Wendet Fault-Objekte ab fault.start_time in der flow()-Methode an.
    """

    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_between, Cout,
                 valves_between, valves_out, h_target,
                 initial_state, faults: List[Fault], initial_t=0.0):
        super().__init__(initial_t, initial_state)
        self.n              = n_tanks
        self.A              = list(A)
        self.Qpmax          = list(Qpmax)
        self.Qf             = Qf
        self.C_between      = list(C_between)
        self.Cout           = list(Cout)
        self.valves_between = [Valve(v.open, v.position) for v in valves_between]
        self.valves_out     = [Valve(v.open, v.position) for v in valves_out]
        self.h_target       = list(h_target)
        self.faults         = faults
        self._applied       = set()   # bereits angewendete Fault-Indizes

    def _apply_faults(self, t: float) -> None:
        """Prüft ob ein Störfall ab Zeitpunkt t aktiv werden soll."""
        for i, fault in enumerate(self.faults):
            if i in self._applied:
                continue
            if t >= fault.start_time:
                self._applied.add(i)
                if fault.fault_type == "valve_stuck":
                    idx = fault.tank_idx
                    if 0 <= idx < len(self.valves_between):
                        self.valves_between[idx].position = fault.magnitude
                        if fault.magnitude == 0.0:
                            self.valves_between[idx].open = False
                elif fault.fault_type == "pump_failure":
                    idx = fault.tank_idx
                    if 0 <= idx < self.n:
                        self.Qpmax[idx] = 0.0
                elif fault.fault_type == "pump_rate_change":
                    idx = fault.tank_idx
                    if 0 <= idx < self.n:
                        self.Qpmax[idx] = fault.magnitude
                elif fault.fault_type == "leak_increase":
                    self.Qf = self.Qf * fault.magnitude

    @override
    def flow(self, t, state):
        self._apply_faults(t)

        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)

        Qp = np.array([
            self.Qpmax[i] if h[i] < self.h_target[i] else 0.0
            for i in range(n)
        ])

        Q_between = np.zeros(max(n - 1, 0))
        for i in range(n - 1):
            eff = self.valves_between[i].effective()
            Q_between[i] = self.C_between[i] * eff * np.sqrt(max(h[i] - h[i + 1], 0.0))

        Q_out = np.zeros(n)
        for i in range(n):
            eff = self.valves_out[i].effective()
            Q_out[i] = self.Cout[i] * eff * np.sqrt(max(h[i], 0.0))

        Q_leak = np.array([self.Qf * np.sqrt(max(h[i], 0.0)) for i in range(n)])

        if n == 1:
            dhdt[0] = (Qp[0] - Q_leak[0] - Q_out[0]) / self.A[0]
        else:
            dhdt[0] = (Qp[0] - Q_leak[0] - Q_between[0] - Q_out[0]) / self.A[0]
            for i in range(1, n - 1):
                dhdt[i] = (Qp[i] + Q_between[i-1] - Q_leak[i]
                           - Q_between[i] - Q_out[i]) / self.A[i]
            dhdt[-1] = (Qp[-1] + Q_between[-1] - Q_leak[-1] - Q_out[-1]) / self.A[-1]

        for i in range(n):
            if h[i] <= 0.0 and dhdt[i] < 0.0:
                dhdt[i] = 0.0
            if h[i] >= self.h_target[i] and dhdt[i] > 0.0:
                dhdt[i] = 0.0

        return dhdt


class NTankFullyCoupledFault(OdeSystem):
    """
    Erweiterte vollvermaschte Tanktopologie mit Störfall-Unterstützung.
    """

    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_all, Cout,
                 valves_between, valves_out, h_target,
                 initial_state, faults: List[Fault], initial_t=0.0):
        super().__init__(initial_t, initial_state)
        self.n              = n_tanks
        self.A              = list(A)
        self.Qpmax          = list(Qpmax)
        self.Qf             = Qf
        self.C_all          = list(C_all)
        self.Cout           = list(Cout)
        self.valves_between = [Valve(v.open, v.position) for v in valves_between]
        self.valves_out     = [Valve(v.open, v.position) for v in valves_out]
        self.h_target       = list(h_target)
        self.faults         = faults
        self._applied       = set()

    def _pair_index(self, i, j):
        n = self.n
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def _apply_faults(self, t: float) -> None:
        for i, fault in enumerate(self.faults):
            if i in self._applied:
                continue
            if t >= fault.start_time:
                self._applied.add(i)
                if fault.fault_type == "valve_stuck":
                    idx = fault.tank_idx
                    if 0 <= idx < len(self.valves_between):
                        self.valves_between[idx].position = fault.magnitude
                        if fault.magnitude == 0.0:
                            self.valves_between[idx].open = False
                elif fault.fault_type == "pump_failure":
                    idx = fault.tank_idx
                    if 0 <= idx < self.n:
                        self.Qpmax[idx] = 0.0
                elif fault.fault_type == "pump_rate_change":
                    idx = fault.tank_idx
                    if 0 <= idx < self.n:
                        self.Qpmax[idx] = fault.magnitude
                elif fault.fault_type == "leak_increase":
                    self.Qf = self.Qf * fault.magnitude

    @override
    def flow(self, t, state):
        self._apply_faults(t)

        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)

        for i in range(n):
            if h[i] < self.h_target[i]:
                dhdt[i] += self.Qpmax[i]
        for i in range(n):
            dhdt[i] -= self.Qf * np.sqrt(max(h[i], 0.0))
        for i in range(n):
            eff = self.valves_out[i].effective()
            dhdt[i] -= self.Cout[i] * eff * np.sqrt(max(h[i], 0.0))
        for i in range(n):
            for j in range(i + 1, n):
                idx = self._pair_index(i, j)
                eff = self.valves_between[idx].effective()
                diff = h[i] - h[j]
                if diff > 0:
                    Q = self.C_all[idx] * eff * np.sqrt(diff)
                    dhdt[i] -= Q
                    dhdt[j] += Q
                elif diff < 0:
                    Q = self.C_all[idx] * eff * np.sqrt(-diff)
                    dhdt[j] -= Q
                    dhdt[i] += Q
        for i in range(n):
            dhdt[i] /= self.A[i]
        for i in range(n):
            if h[i] <= 0.0 and dhdt[i] < 0.0:
                dhdt[i] = 0.0
            if h[i] >= self.h_target[i] and dhdt[i] > 0.0:
                dhdt[i] = 0.0

        return dhdt


# ══════════════════════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def build_fault_system(topology, n_tanks, valves_between, valves_out,
                       qp_list, h_target, areas, faults: List[Fault]):
    """Erzeugt ein fault-fähiges ODE-System je nach Topologie."""
    initial_state = NTankState(h=[0.0] * n_tanks)
    if topology == "linear":
        return NTankLinearFault(
            n_tanks=n_tanks, A=areas,
            Qpmax=qp_list, Qf=0.01,
            C_between=[0.01] * (n_tanks - 1),
            Cout=[0.01] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            h_target=h_target,
            initial_state=initial_state,
            faults=faults,
        )
    else:
        n_between = n_tanks * (n_tanks - 1) // 2
        return NTankFullyCoupledFault(
            n_tanks=n_tanks, A=areas,
            Qpmax=qp_list, Qf=0.01,
            C_all=[0.01] * n_between,
            Cout=[0.01] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            h_target=h_target,
            initial_state=initial_state,
            faults=faults,
        )


def simulate(topology, n_tanks, valves_between, valves_out,
             qp_list, h_target, areas, n_samples,
             faults: Optional[List[Fault]] = None) -> pl.DataFrame:
    """Führt eine Simulation durch und gibt den DataFrame zurück."""
    collector = FrameCollector(n_tanks, noise_std=0.0)
    system = build_fault_system(
        topology, n_tanks, valves_between, valves_out,
        qp_list, h_target, areas, faults or []
    )
    env = OdeEnvironment(system, dt=1.0,
                         map_to_dataframe=collector.collect_frame)
    collect(env, n_samples)
    return collector.concat()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT-FUNKTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_fault_comparison(df_normal: pl.DataFrame, df_fault: pl.DataFrame,
                          n_tanks: int, fault: Fault,
                          title: str, filename: str) -> None:
    """
    Vergleicht Normal- vs. Störfall-Verlauf für alle Tanks.
    Markiert den Störfall-Zeitpunkt als vertikale rote Linie.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f"]

    fig, axes = plt.subplots(n_tanks, 1,
                             figsize=(11, 3 * n_tanks), sharex=True)
    if n_tanks == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i in range(n_tanks):
        col      = f"h{i + 1}"
        t        = df_normal["t"].to_numpy()
        h_normal = df_normal[col].to_numpy()
        h_fault  = df_fault[col].to_numpy()
        # ODE-Solver kann minimal unterschiedlich viele Schritte erzeugen
        # → auf gemeinsame Länge kürzen
        n_min    = min(len(t), len(h_normal), len(h_fault))
        t        = t[:n_min]
        h_normal = h_normal[:n_min]
        h_fault  = h_fault[:n_min]
        c        = colors[i % len(colors)]

        axes[i].plot(t, h_normal, color=c, linewidth=2.2, alpha=0.55,
                     label="Normal", linestyle="-")
        axes[i].plot(t, h_fault, color="#000000", linewidth=2.0,
                     label="Störfall", linestyle="--", alpha=0.95)
        axes[i].axvline(fault.start_time, color="red", linewidth=2.0,
                        linestyle=":", label=f"Störfall bei t={fault.start_time}s")
        axes[i].set_ylabel("Füllstand h [m]", fontsize=9)
        axes[i].set_title(f"Tank {i + 1}", fontsize=10)
        axes[i].legend(fontsize=8, loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Zeit t [s]", fontsize=10)
    plt.tight_layout()

    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {filename} gespeichert")


# ══════════════════════════════════════════════════════════════════════════════
#  HAUPTFUNKTION: ALLE SZENARIEN AUSFÜHREN
# ══════════════════════════════════════════════════════════════════════════════

def faults_from_config(cfg: dict) -> List[Fault]:
    """
    Liest die 'faults' Liste aus der config.yaml und gibt Fault-Objekte zurück.

    config.yaml Beispiel:
      faults:
        - {type: "valve_stuck",   time: 200, idx: 0, magnitude: 0.5}
        - {type: "pump_failure",  time: 400, idx: 0, magnitude: 0.0}
        - {type: "leak_increase", time: 450, idx: 0, magnitude: 20.0}

      Kein Störfall:
        faults: []
    """
    raw = cfg.get("faults", [])
    if not raw:
        return []
    faults = []
    for entry in raw:
        faults.append(Fault(
            fault_type = entry["type"],
            start_time = float(entry["time"]),
            tank_idx   = int(entry.get("idx", 0)),
            magnitude  = float(entry.get("magnitude", 0.0)),
        ))
    return faults


def run_fault_scenarios(
    topology:   str,
    n_tanks:    int,
    qp_list:    List[float],
    h_target:   List[float],
    areas:      List[float],
    faults:     List[Fault],
    n_samples:  int = 500,
) -> None:
    """
    Führt Störfall-Szenarien durch und speichert Vergleichs-Plots.

    Jeder Fault in der Liste bekommt einen eigenen Plot (Normal vs. Störfall).
    Die faults-Liste kommt direkt aus config.yaml über faults_from_config().

    Wenn faults leer ist → wird diese Funktion übersprungen.

    Beispiel config.yaml:
      faults:
        - {type: "valve_stuck",   time: 425, idx: 0, magnitude: 0.0}
        - {type: "pump_failure",  time: 425, idx: 1, magnitude: 0.0}
        - {type: "leak_increase", time: 425, idx: 0, magnitude: 20.0}

      Kein Störfall:
        faults: []
    """
    # ── Kein Störfall definiert → überspringen ────────────────────────────────
    if not faults:
        print("\n  Keine Störfälle in config.yaml definiert → übersprungen")
        return

    print(f"\n{'═' * 55}")
    print("  Störfall-Szenarien (Kapitel 4.2)")
    print(f"{'═' * 55}")
    print(f"  Topologie : {topology} | Tanks: {n_tanks} | Samples: {n_samples}")
    print(f"  Störfälle : {len(faults)} definiert")
    for i, f in enumerate(faults):
        print(f"    [{i+1}] {f.fault_type:15s} | t={f.start_time}s "
              f"| idx={f.tank_idx} | magnitude={f.magnitude}")
    print(f"{'─' * 55}\n")

    n_between      = (n_tanks - 1) if topology == "linear" else n_tanks * (n_tanks - 1) // 2
    valves_between = [Valve(open=True, position=1.0) for _ in range(n_between)]
    valves_out     = [Valve(open=True, position=1.0) for _ in range(n_tanks)]
    topo_label     = "Linear" if topology == "linear" else "Coupled"

    # ── Normalsimulation (einmal, Referenz für alle Szenarien) ────────────────
    print("Simuliere Normalverlauf ...")
    df_normal = simulate(topology, n_tanks, valves_between, valves_out,
                         qp_list, h_target, areas, n_samples)

    # ── Pro Fault: eigene Störfall-Simulation + Plot ───────────────────────────
    for i, fault in enumerate(faults):
        label = {
            "valve_stuck":   "Ventilausfall",
            "pump_failure":  "Pumpenausfall",
            "leak_increase": "Leckage-Erhöhung",
        }.get(fault.fault_type, fault.fault_type)

        print(f"\n[{i+1}/{len(faults)}] Szenario: {label}")

        df_fault = simulate(topology, n_tanks, valves_between, valves_out,
                            qp_list, h_target, areas, n_samples,
                            faults=[fault])

        # Beschriftung je nach Typ
        if fault.fault_type == "valve_stuck":
            subtitle = (f"Ventil {fault.tank_idx} → Position {fault.magnitude} "
                        f"bei t={fault.start_time}s")
        elif fault.fault_type == "pump_failure":
            subtitle = f"Pumpe Tank {fault.tank_idx + 1} fällt bei t={fault.start_time}s aus"
        else:
            subtitle = (f"Leckagekoeffizient ×{fault.magnitude} "
                        f"bei t={fault.start_time}s")

        plot_fault_comparison(
            df_normal, df_fault, n_tanks, fault,
            title=f"Szenario {i+1}: {label} – {topo_label}, {n_tanks} Tanks\n{subtitle}",
            filename=f"fault_{i+1}_{fault.fault_type}_{topology}_{n_tanks}tanks.png",
        )

    print(f"\n✓ Alle {len(faults)} Störfall-Szenarien abgeschlossen")
    print(f"  Gespeichert in: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE-EINSTIEG (direkt via python fault_scenarios.py)
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    flowcean.cli.initialize()
    initialize_random(seed=42)

    run_fault_scenarios(
        topology  = cfg["topology"],
        n_tanks   = cfg["n_tanks"],
        qp_list   = cfg["pumps"],
        h_target  = cfg["h_target"],
        areas     = cfg["areas"],
        faults    = faults_from_config(cfg),
        n_samples = cfg.get("n_samples_fault", 500),
    )


if __name__ == "__main__":
    main()