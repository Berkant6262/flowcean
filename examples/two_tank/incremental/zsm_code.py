# ═══════════════════════════════════════════════════════════════════════════════
#  N-Tank-Simulation mit inkrementellem Machine Learning
#  -----------------------------------------------------------------------
#  Dieses Skript simuliert ein N-Tank-System als ODE-Modell und trainiert
#  einen River-Lerner (HoeffdingTree oder MLP), der die Füllstandsdynamik
#  vorhersagt. Zwei Topologien werden unterstützt:
#    • Linear   – Tanks in Reihe (Kaskadenfluss)
#    • Coupled  – Jeder Tank ist mit jedem anderen verbunden (Vollvermaschung)
#
#  Abhängigkeiten: flowcean, river, polars, numpy, matplotlib
# ═══════════════════════════════════════════════════════════════════════════════

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray
from river import neural_net, optim, tree
from typing_extensions import Self, override

import flowcean.cli
from flowcean.core import evaluate_offline, learn_incremental
from flowcean.ode import OdeEnvironment, OdeState, OdeSystem
from flowcean.polars import SlidingWindow, StreamingOfflineEnvironment, TrainTestSplit
from flowcean.polars.environments.dataframe import collect
from flowcean.river import RiverLearner
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError

# Der HoeffdingTree nutzt intern Rekursion bei tiefen Bäumen.
# max_depth=5 begrenzt die Tiefe, aber 10 000 bleibt als Sicherheitsnetz,
# um RecursionErrors bei komplexen Daten zu vermeiden.
sys.setrecursionlimit(10000)

# Modul-Logger – Meldungen erscheinen nur, wenn Logging konfiguriert ist.
logger = logging.getLogger(__name__)

# Ausgabepfad für alle generierten Plots (aktuelles Arbeitsverzeichnis).
OUTPUT_DIR = Path.cwd()


# ─────────────────────────────────────────
#  Datenstrukturen
# ─────────────────────────────────────────

@dataclass
class Valve:
    """
    Repräsentiert ein einzelnes Ventil im Tanksystem.

    Attribute:
        open     – True = Ventil geöffnet, False = geschlossen (kein Durchfluss).
        position – Öffnungsgrad zwischen 0.0 (vollständig zu) und 1.0 (vollständig auf).
    """
    open: bool = True
    position: float = 1.0

    def effective(self) -> float:
        """
        Gibt den effektiven Öffnungsgrad zurück.
        Ist das Ventil geschlossen, wird unabhängig von position 0.0 zurückgegeben.
        Dieser Wert fließt direkt als Multiplikator in die Durchflussberechnung ein.
        """
        return self.position if self.open else 0.0


@dataclass
class NTankState(OdeState):
    """
    Zustandsvektor des N-Tank-Systems.

    h – Liste der aktuellen Füllstände [m] für jeden Tank.
    Erbt von OdeState (flowcean-Basisklasse) und implementiert die
    Konvertierung zu/von NumPy-Arrays für den ODE-Solver.
    """
    h: List[float]

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        """Konvertiert die Füllstandsliste in ein NumPy-Array (benötigt vom ODE-Solver)."""
        return np.array(self.h, dtype=np.float64)

    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        """Rekonstruiert einen NTankState aus einem NumPy-Array (nach jedem ODE-Schritt)."""
        return cls(state.tolist())


# ─────────────────────────────────────────
#  Frame-Collector
# ─────────────────────────────────────────

class FrameCollector:
    """
    Sammelt ODE-Ausgaben schrittweise als Polars-DataFrames und
    ermöglicht optional das Überlagern mit Gaußschem Sensorrauschen.

    Wird als Callback an OdeEnvironment übergeben: Bei jedem Zeitschritt
    ruft flowcean collect_frame(ts, xs) auf.
    """

    def __init__(self, n_tanks: int, noise_std: float = 0.0) -> None:
        """
        Args:
            n_tanks   – Anzahl der Tanks (bestimmt Spaltenanzahl).
            noise_std – Standardabweichung des Gaußschen Rauschens in Metern.
                        0.0 = kein Rauschen (ideale Sensoren).
        """
        self.n_tanks = n_tanks
        self.noise_std = noise_std
        self.frames: List[pl.DataFrame] = []  # Puffer aller gesammelten Frames

    def clear(self) -> None:
        """Leert den internen Frame-Puffer (z. B. für einen Neustart)."""
        self.frames.clear()

    def collect_frame(self, ts, xs) -> pl.DataFrame:
        """
        Callback-Funktion für OdeEnvironment.
        Wandelt einen ODE-Zeitschritt in einen Polars-DataFrame um und
        hängt ihn an den internen Puffer an.

        Args:
            ts – Liste von Zeitstempeln des aktuellen Schritts.
            xs – Liste von NTankState-Objekten (ein Zustand je Zeitstempel).

        Returns:
            pl.DataFrame mit Spalten: t, h1, h2, ..., hN
        """
        frame = pl.DataFrame({
            "t": ts,
            **{
                f"h{i+1}": [
                    # Füllstand darf nicht negativ werden (physikalische Untergrenze).
                    # Optional: Gaußsches Rauschen simuliert Sensorfehler.
                    max(0.0, x.h[i] + (
                        np.random.normal(0, self.noise_std)
                        if self.noise_std > 0 else 0.0
                    ))
                    for x in xs
                ]
                for i in range(self.n_tanks)
            },
        })
        self.frames.append(frame)
        return frame

    def concat(self) -> pl.DataFrame:
        """Fügt alle gesammelten Frames zu einem einzigen DataFrame zusammen."""
        return pl.concat(self.frames)


# ─────────────────────────────────────────
#  Topologie 1: Linear (Kette)
# ─────────────────────────────────────────

class NTankLinear(OdeSystem[NTankState]):
    """
    ODE-System für eine lineare (kaskadierende) Tanktopologie.

    Physikalisches Modell:
      • Tank 1 → Tank 2 → ... → Tank N (Schwerkraftfluss nur in eine Richtung).
      • Jeder Tank hat eine eigene Pumpe (Qpmax), die stoppt, sobald h >= h_target.
      • Zwischenfluss: Q_between[i] = C_between[i] * eff * sqrt(max(h[i] - h[i+1], 0))
      • Auslass:       Q_out[i]     = Cout[i] * eff * sqrt(max(h[i], 0))
      • Leckage:       Q_leak[i]    = Qf * sqrt(max(h[i], 0))

    Schutzmechanismen:
      • Unterlauf-Schutz: dhdt wird auf 0 gesetzt, wenn h <= 0 und dhdt < 0.
      • Überlauf-Schutz:  dhdt wird auf 0 gesetzt, wenn h >= h_target und dhdt > 0.
    """

    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_between, Cout,
                 valves_between, valves_out, h_target,
                 initial_state, initial_t=0.0):
        """
        Args:
            n_tanks        – Anzahl der Tanks.
            A              – Querschnittsflächen [m²] pro Tank (Liste).
            Qpmax          – Maximale Pumpenleistung [m³/s] pro Tank.
            Qf             – Leckage-Koeffizient (global für alle Tanks).
            C_between      – Durchflusskoeffizienten zwischen benachbarten Tanks.
            Cout           – Auslass-Durchflusskoeffizienten pro Tank.
            valves_between – Valve-Objekte für Verbindungen Tank i → Tank i+1.
            valves_out     – Valve-Objekte für Auslässe.
            h_target       – Ziel-Füllstände [m] (Pumpe schaltet ab + Überlaufgrenze).
            initial_state  – Startzustand als NTankState.
            initial_t      – Startzeitpunkt (Standard: 0.0 s).
        """
        super().__init__(initial_t, initial_state)
        self.n = n_tanks
        self.A = A
        self.Qpmax = Qpmax
        self.Qf = Qf
        self.C_between = C_between
        self.Cout = Cout
        self.valves_between = valves_between
        self.valves_out = valves_out
        self.h_target = h_target

    @override
    def flow(self, t, state):
        """
        Berechnet die Ableitungen dhdt für jeden Tank zum Zeitpunkt t.
        Wird vom ODE-Solver (z. B. RK45) bei jedem Integrationsschritt aufgerufen.

        Bilanzgleichung pro Tank i:
          A[i] * dh[i]/dt = Qp[i] + Q_between_in[i]
                           - Q_between_out[i] - Q_out[i] - Q_leak[i]
        """
        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)

        # Pumpe läuft nur, solange Füllstand unter Zielwert liegt.
        Qp = np.array([
            self.Qpmax[i] if h[i] < self.h_target[i] else 0.0
            for i in range(n)
        ])

        # Schwerkraftfluss nur vom höheren zum niedrigeren Tank (Kaskade).
        # sqrt(Δh) modelliert den Torricelli-ähnlichen Druckabfall.
        Q_between = np.zeros(max(n - 1, 0))
        for i in range(n - 1):
            eff = self.valves_between[i].effective()
            Q_between[i] = self.C_between[i] * eff * np.sqrt(max(h[i] - h[i + 1], 0.0))

        # Auslass jedes Tanks (Schwerkraftentleerung).
        Q_out = np.zeros(n)
        for i in range(n):
            eff = self.valves_out[i].effective()
            Q_out[i] = self.Cout[i] * eff * np.sqrt(max(h[i], 0.0))

        # Konstante Leckage proportional zur Wurzel des Füllstands.
        Q_leak = np.array([self.Qf * np.sqrt(max(h[i], 0.0)) for i in range(n)])

        # Differentialgleichungen aufstellen (Massenerhaltung / Kontinuität).
        if n == 1:
            dhdt[0] = (Qp[0] - Q_leak[0] - Q_out[0]) / self.A[0]
        else:
            # Erster Tank: kein Zufluss von vorherigem Tank.
            dhdt[0] = (Qp[0] - Q_leak[0] - Q_between[0] - Q_out[0]) / self.A[0]
            # Mittlere Tanks: erhalten Zufluss vom vorherigen Tank.
            for i in range(1, n - 1):
                dhdt[i] = (Qp[i] + Q_between[i - 1] - Q_leak[i] - Q_between[i] - Q_out[i]) / self.A[i]
            # Letzter Tank: kein Abfluss zu nachfolgendem Tank.
            dhdt[-1] = (Qp[-1] + Q_between[-1] - Q_leak[-1] - Q_out[-1]) / self.A[-1]

        # Schutzmechanismen gegen physikalisch unmögliche Zustände.
        for i in range(n):
            # Unterlauf-Schutz: Tank kann nicht unter 0 m fallen.
            if h[i] <= 0.0 and dhdt[i] < 0.0:
                dhdt[i] = 0.0
            # Überlauf-Schutz: h_target ist die physikalische Obergrenze.
            if h[i] >= self.h_target[i] and dhdt[i] > 0.0:
                dhdt[i] = 0.0

        return dhdt


# ─────────────────────────────────────────
#  Topologie 2: Vollvermascht
# ─────────────────────────────────────────

class NTankFullyCoupled(OdeSystem[NTankState]):
    """
    ODE-System für eine vollvermaschte Tanktopologie.

    Physikalisches Modell:
      • Jedes Tankpaar (i, j) ist bidirektional verbunden.
      • Fluss folgt dem Druckgefälle: von höherem zu niedrigerem Tank.
      • Anzahl der Verbindungen: n*(n-1)/2 (untere Dreiecksmatrix).
      • Paar-Index wird über _pair_index(i, j) berechnet (komprimiertes Mapping).
    """

    def __init__(self, *, n_tanks, A, Qpmax, Qf, C_all, Cout,
                 valves_between, valves_out, h_target,
                 initial_state, initial_t=0.0):
        """
        Args:
            C_all – Durchflusskoeffizienten für alle n*(n-1)/2 Tankpaare.
                    Reihenfolge: (0,1), (0,2), ..., (0,n-1), (1,2), ...
            Alle anderen Parameter: siehe NTankLinear.
        """
        super().__init__(initial_t, initial_state)
        self.n = n_tanks
        self.A = A
        self.Qpmax = Qpmax
        self.Qf = Qf
        self.C_all = C_all
        self.Cout = Cout
        self.valves_between = valves_between
        self.valves_out = valves_out
        self.h_target = h_target

    def _pair_index(self, i, j):
        """
        Berechnet den flachen Index für das Tankpaar (i, j) mit i < j.
        Bildet die obere Dreiecksmatrix auf eine 1D-Liste ab.
        Formel: i*(2n - i - 1)/2 + (j - i - 1)
        """
        n = self.n
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    @override
    def flow(self, t, state):
        """
        Berechnet dhdt für alle Tanks in der vollvermaschten Topologie.
        Iteriert über alle eindeutigen Paare (i < j) und verteilt den
        Fluss symmetrisch: dhdt[i] -= Q, dhdt[j] += Q (oder umgekehrt).
        """
        h = state.astype(float)
        n = self.n
        dhdt = np.zeros_like(h)

        # Pumpen-Beitrag: Zufluss nur wenn Füllstand unter Zielwert.
        for i in range(n):
            if h[i] < self.h_target[i]:
                dhdt[i] += self.Qpmax[i]

        # Leckage-Abzug für alle Tanks.
        for i in range(n):
            dhdt[i] -= self.Qf * np.sqrt(max(h[i], 0.0))

        # Auslass-Abzug für alle Tanks.
        for i in range(n):
            eff = self.valves_out[i].effective()
            dhdt[i] -= self.Cout[i] * eff * np.sqrt(max(h[i], 0.0))

        # Bidirektionaler Fluss zwischen allen Tankpaaren.
        # Richtung wird durch das Vorzeichen der Füllstandsdifferenz bestimmt.
        for i in range(n):
            for j in range(i + 1, n):
                idx = self._pair_index(i, j)
                eff = self.valves_between[idx].effective()
                diff = h[i] - h[j]
                if diff > 0:
                    # Fluss von Tank i nach Tank j.
                    Q = self.C_all[idx] * eff * np.sqrt(diff)
                    dhdt[i] -= Q
                    dhdt[j] += Q
                elif diff < 0:
                    # Fluss von Tank j nach Tank i.
                    Q = self.C_all[idx] * eff * np.sqrt(-diff)
                    dhdt[j] -= Q
                    dhdt[i] += Q

        # Division durch Querschnittsfläche (Kontinuitätsgleichung: dh/dt = Q/A).
        for i in range(n):
            dhdt[i] /= self.A[i]

        # Schutzmechanismen (identisch zur linearen Topologie).
        for i in range(n):
            if h[i] <= 0.0 and dhdt[i] < 0.0:
                dhdt[i] = 0.0
            if h[i] >= self.h_target[i] and dhdt[i] > 0.0:
                dhdt[i] = 0.0

        return dhdt


# ─────────────────────────────────────────
#  Ziel-Füllstand konfigurieren
# ─────────────────────────────────────────

def configure_h_target(n_tanks: int) -> List[float]:
    """
    Interaktive Konfiguration der Ziel-Füllstände h_target pro Tank.

    h_target erfüllt zwei Rollen gleichzeitig:
      1. Pumpen-Schwellwert: Pumpe schaltet ab, wenn h >= h_target.
      2. Physikalische Obergrenze: kein Zufluss kann h über h_target treiben.

    Rückgabe:
        Liste mit h_target-Werten [m] für jeden Tank.
    """
    print("\n── Ziel-Füllstände pro Tank ──────────────────")
    print("  (Pumpe stoppt & Tank kann nicht überlaufen sobald h[i] >= h_target[i])")

    use_same = input(
        "  Gleichen Zielwert für alle Tanks? (j/n) [Standard: j]: "
    ).strip().lower() != "n"

    if use_same:
        while True:
            raw = input("  Ziel-Füllstand für alle Tanks (m) [Standard: 0.5]: ").strip()
            if raw == "":
                val = 0.5
                break
            try:
                val = float(raw)
                if val > 0:
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"  → Alle Tanks: h_target = {val} m")
        return [val] * n_tanks

    h_target = []
    for i in range(n_tanks):
        while True:
            raw = input(f"  Tank {i + 1} – Ziel-Füllstand (m) [Standard: 0.5]: ").strip()
            if raw == "":
                h_target.append(0.5)
                break
            try:
                val = float(raw)
                if val > 0:
                    h_target.append(val)
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"    → Tank {i + 1}: h_target = {h_target[-1]} m")

    return h_target


def print_h_target_summary(h_target: List[float]) -> None:
    """Gibt eine übersichtliche Zusammenfassung der konfigurierten Ziel-Füllstände aus."""
    print("\n── Ziel-Füllstand-Übersicht ─────────────────")
    for i, ht in enumerate(h_target):
        print(f"  Tank {i + 1}: h_target = {ht:.3f} m")
    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────
#  Tankgröße konfigurieren
# ─────────────────────────────────────────

def configure_tank_areas(n_tanks: int) -> List[float]:
    """
    Interaktive Konfiguration der Querschnittsflächen A [m²] pro Tank.

    A beeinflusst die Systemdynamik direkt: dh/dt = Q_netto / A.
      • Großes A → träge Reaktion (Füllstand ändert sich langsam).
      • Kleines A → schnelle Reaktion (Füllstand ändert sich schnell).

    Standardwert 0.0154 m² ≈ kreisförmiger Tank mit ~14 cm Durchmesser.
    """
    print("\n── Tankgröße (Querschnittsfläche A) ─────────")
    print("  (Größeres A = träger, kleineres A = reaktiver)")

    use_same = input(
        "  Gleiche Tankgröße für alle? (j/n) [Standard: j]: "
    ).strip().lower() != "n"

    if use_same:
        while True:
            raw = input("  A für alle Tanks (m²) [Standard: 0.0154]: ").strip()
            if raw == "":
                val = 0.0154
                break
            try:
                val = float(raw)
                if val > 0:
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"  → Alle Tanks: A = {val} m²")
        return [val] * n_tanks

    areas = []
    for i in range(n_tanks):
        while True:
            raw = input(f"  Tank {i + 1} – A (m²) [Standard: 0.0154]: ").strip()
            if raw == "":
                areas.append(0.0154)
                break
            try:
                val = float(raw)
                if val > 0:
                    areas.append(val)
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"    → Tank {i + 1}: A = {areas[-1]} m²")
    return areas


def print_tank_areas_summary(areas: List[float]) -> None:
    """Gibt eine übersichtliche Zusammenfassung der konfigurierten Tankflächen aus."""
    print("\n── Tankgröße-Übersicht ──────────────────────")
    for i, a in enumerate(areas):
        print(f"  Tank {i + 1}: A = {a:.6f} m²")
    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────
#  Rauschen konfigurieren
# ─────────────────────────────────────────

def configure_noise() -> float:
    """
    Interaktive Konfiguration des Gaußschen Sensorrauschens.

    Das Rauschen N(0, σ²) wird bei der Datenerfassung zu jedem Füllstandswert
    addiert und simuliert reale Messungenauigkeiten (z. B. Ultraschall-Sensoren).
    Das ML-Modell muss damit robuster trainiert werden.

    Rückgabe:
        σ (Standardabweichung) in Metern. 0.0 = ideale Sensoren.
    """
    print("\n── Sensor-Rauschen ───────────────────────────")
    use_noise = input(
        "  Rauschen aktivieren? (j/n) [Standard: n]: "
    ).strip().lower() == "j"

    if not use_noise:
        print("  → Kein Rauschen")
        return 0.0

    while True:
        raw = input("  Standardabweichung σ (m) [Standard: 0.01]: ").strip()
        if raw == "":
            sigma = 0.01
            break
        try:
            sigma = float(raw)
            if sigma > 0:
                break
            print("  ⚠ Wert muss größer als 0 sein.")
        except ValueError:
            print("  ⚠ Bitte eine Zahl eingeben.")

    print(f"  → Rauschen aktiv: σ = {sigma} m")
    return sigma


# ─────────────────────────────────────────
#  Pumpen-Konfiguration
# ─────────────────────────────────────────

def configure_pumps(n_tanks: int) -> List[float]:
    """
    Interaktive Konfiguration der maximalen Pumpenleistung Qpmax [m³/s] pro Tank.

    Tanks ohne Pumpe (Qpmax = 0) füllen sich ausschließlich durch Zufluss von
    benachbarten Tanks. Dies ermöglicht das Modellieren von passiven Tanks.

    Rückgabe:
        Liste mit Qpmax-Werten. 0.0 = kein Pump an diesem Tank.
    """
    pump_mode = prompt_choice(
        "\nPumpen-Modus – 'alle' (jeder Tank) oder 'manuell' (auswählen): ",
        ("alle", "manuell"),
    )

    if pump_mode == "alle":
        while True:
            raw = input("  Qpmax für alle Tanks (m³/s) [Standard: 0.01]: ").strip()
            if raw == "":
                qp = 1e-2
                break
            try:
                qp = float(raw)
                if qp > 0:
                    break
                print("  ⚠ Wert muss größer als 0 sein.")
            except ValueError:
                print("  ⚠ Bitte eine Zahl eingeben.")
        print(f"  → Alle {n_tanks} Tanks: Pumpe aktiv (Qpmax={qp})")
        return [qp] * n_tanks

    qp_list = []
    for i in range(n_tanks):
        has_pump = input(f"  Tank {i + 1} – Pumpe? (j/n) [Standard: j]: ").strip().lower() != "n"
        if has_pump:
            while True:
                raw = input(f"    Tank {i + 1} – Qpmax (m³/s) [Standard: 0.01]: ").strip()
                if raw == "":
                    qp = 1e-2
                    break
                try:
                    qp = float(raw)
                    if qp > 0:
                        break
                    print("    ⚠ Wert muss größer als 0 sein.")
                except ValueError:
                    print("    ⚠ Bitte eine Zahl eingeben.")
            qp_list.append(qp)
            print(f"    → Tank {i + 1}: Pumpe aktiv (Qpmax={qp})")
        else:
            qp_list.append(0.0)
            print(f"    → Tank {i + 1}: keine Pumpe")
    return qp_list


def print_pump_summary(qp_list: List[float]) -> None:
    """Gibt eine übersichtliche Zusammenfassung der Pumpen-Konfiguration aus."""
    print("\n── Pumpen-Übersicht ─────────────────────────")
    for i, qp in enumerate(qp_list):
        status = f"aktiv (Qpmax={qp})" if qp > 0 else "KEINE PUMPE"
        print(f"  Tank {i + 1}: {status}")
    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────
#  System-Instanz erstellen
# ─────────────────────────────────────────

def build_system(topology: str, n_tanks: int,
                 valves_between: List[Valve], valves_out: List[Valve],
                 qp_list: List[float], h_target: List[float],
                 areas: List[float]):
    """
    Factory-Funktion: Erzeugt die passende ODE-System-Instanz je nach Topologie.

    Alle Tanks starten bei h=0.0 m (leere Tanks).
    Physikalische Standardwerte:
      • C_between = 1.5938e-4 m^(5/2)/s  (Verbindungskoeffizient linear)
      • C_all     = 1.5e-4    m^(5/2)/s  (Verbindungskoeffizient coupled)
      • Cout      = 1.5964e-4 m^(5/2)/s  (Auslasskoeffizient)
      • Qf        = 1e-4      m^(5/2)/s  (Leckagekoeffizient)

    Args:
        topology – "linear" oder "coupled".
        Alle anderen Parameter: direkt aus der Benutzerkonfiguration.
    """
    initial_state = NTankState(h=[0.0] * n_tanks)
    if topology == "linear":
        return NTankLinear(
            n_tanks=n_tanks, A=areas,
            Qpmax=qp_list, Qf=1e-4,
            C_between=[1.5938e-4] * (n_tanks - 1),
            Cout=[1.59640e-4] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            h_target=h_target,
            initial_state=initial_state,
        )
    else:
        # Anzahl der Verbindungen bei vollständiger Vermaschung: n*(n-1)/2
        n_between = n_tanks * (n_tanks - 1) // 2
        return NTankFullyCoupled(
            n_tanks=n_tanks, A=areas,
            Qpmax=qp_list, Qf=1e-4,
            C_all=[1.5e-4] * n_between,
            Cout=[1.59640e-4] * n_tanks,
            valves_between=valves_between,
            valves_out=valves_out,
            h_target=h_target,
            initial_state=initial_state,
        )


# ─────────────────────────────────────────
#  Ventil-Konfiguration
# ─────────────────────────────────────────

def configure_valve(label: str) -> Valve:
    """
    Interaktive Konfiguration eines einzelnen Ventils.

    Fragt zuerst nach dem Öffnungsstatus (offen/geschlossen), dann
    – wenn offen – nach dem Öffnungsgrad (0.0 = vollständig zu, 1.0 = voll offen).

    Args:
        label – Bezeichnung des Ventils für die Konsolenausgabe.
    """
    print(f"\n  Ventil {label}")
    is_open = input("    Geöffnet? (j/n) [Standard: j]: ").strip().lower() != "n"
    if is_open:
        while True:
            raw = input("    Öffnungsgrad (0.0–1.0) [Standard: 1.0]: ").strip()
            if raw == "":
                position = 1.0
                break
            try:
                position = float(raw)
                if 0.0 <= position <= 1.0:
                    break
                print("    ⚠ Wert muss zwischen 0.0 und 1.0 liegen.")
            except ValueError:
                print("    ⚠ Ungültige Eingabe, bitte eine Zahl eingeben.")
    else:
        position = 0.0
    return Valve(open=is_open, position=position)


def configure_valves_between_linear(n_tanks: int) -> List[Valve]:
    """Konfiguriert die n-1 Verbindungsventile der linearen Topologie (Tank i → Tank i+1)."""
    print("\n── Zwischenventile (Tank i → Tank i+1) ──")
    return [configure_valve(f"Tank {i + 1} → Tank {i + 2}") for i in range(n_tanks - 1)]


def configure_valves_between_coupled(n_tanks: int) -> List[Valve]:
    """Konfiguriert alle n*(n-1)/2 Verbindungsventile der vollvermaschten Topologie."""
    print("\n── Zwischenventile (alle Paare) ──")
    valves = []
    for i in range(n_tanks):
        for j in range(i + 1, n_tanks):
            valves.append(configure_valve(f"Tank {i + 1} ↔ Tank {j + 1}"))
    return valves


def configure_valves_out(n_tanks: int) -> List[Valve]:
    """Konfiguriert die n Auslassventile (je einen pro Tank)."""
    print("\n── Auslassventile ──")
    return [configure_valve(f"Auslass Tank {i + 1}") for i in range(n_tanks)]


def print_valve_summary(valves_between: List[Valve], valves_out: List[Valve],
                        topology: str, n_tanks: int) -> None:
    """Gibt eine strukturierte Übersicht aller konfigurierten Ventilstellungen aus."""
    print("\n── Ventil-Übersicht ─────────────────────────")
    if topology == "linear":
        for i, v in enumerate(valves_between):
            print(f"  Zwischenventil Tank {i + 1}→{i + 2}: "
                  f"{'offen' if v.open else 'ZU'}, Position={v.position:.2f}")
    else:
        idx = 0
        for i in range(n_tanks):
            for j in range(i + 1, n_tanks):
                v = valves_between[idx]
                print(f"  Zwischenventil Tank {i + 1}↔{j + 1}: "
                      f"{'offen' if v.open else 'ZU'}, Position={v.position:.2f}")
                idx += 1
    for i, v in enumerate(valves_out):
        print(f"  Auslassventil  Tank {i + 1}: "
              f"{'offen' if v.open else 'ZU'}, Position={v.position:.2f}")
    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────
#  Modell-Auswahl
# ─────────────────────────────────────────

def build_learner(model_choice: str) -> RiverLearner:
    """
    Erzeugt den gewählten inkrementellen River-Lerner als flowcean-RiverLearner.

    Zwei Optionen:
      • "hoeffding" – HoeffdingTreeRegressor:
          Entscheidungsbaum für Datenströme. Lernt schrittweise durch statistischen
          Test (Hoeffding-Bound), wann eine neue Verzweigung gerechtfertigt ist.
          grace_period=50: mindestens 50 Samples bevor ein Split geprüft wird.
          max_depth=5: begrenzt Rekursionstiefe und verhindert Overfitting.

      • "mlp" – MLPRegressor (river):
          Neuronales Netz mit 2 versteckten Schichten (32, 16 Neuronen).
          Aktivierungen: ReLU → ReLU → Identity (lineare Ausgabe für Regression).
          Adam-Optimierer mit Lernrate 1e-3, Seed 42 für Reproduzierbarkeit.

    Args:
        model_choice – "hoeffding" oder "mlp".
    """
    if model_choice == "mlp":
        return RiverLearner(
            model=neural_net.MLPRegressor(
                hidden_dims=(32, 16),
                activations=(
                    neural_net.activations.ReLU,
                    neural_net.activations.ReLU,
                    neural_net.activations.Identity,
                ),
                optimizer=optim.Adam(lr=1e-3),
                seed=42,
            )
        )
    else:
        return RiverLearner(
            model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5)
        )


# ─────────────────────────────────────────
#  River-Modell sicher extrahieren
# ─────────────────────────────────────────

def extract_river_model(trained_model):
    """
    Robuster Zugriff auf das zugrunde liegende River-Modell aus dem flowcean-Wrapper.

    learn_incremental gibt je nach flowcean-Version ein gewrapptes Objekt zurück.
    Die Attributnamen variieren zwischen Versionen (model, _model, learner, _learner).
    Diese Funktion probiert alle bekannten Attributnamen der Reihe nach durch und
    gibt das erste gefundene zurück. Falls keines existiert, wird trained_model
    direkt verwendet (Fallback mit Logger-Warnung).
    """
    for attr in ("model", "_model", "learner", "_learner"):
        candidate = getattr(trained_model, attr, None)
        if candidate is not None:
            return candidate

    logger.warning(
        "Kein bekanntes Wrapper-Attribut gefunden. "
        "Nutze trained_model direkt als River-Modell."
    )
    return trained_model


# ─────────────────────────────────────────
#  Metriken aus Report lesen
# ─────────────────────────────────────────

def extract_metric(report_obj, metric_name: str, target: str) -> float:
    """
    Extrahiert einen numerischen Metrikwert aus dem flowcean-Evaluierungsbericht.

    Zweistufige Strategie:
      1. Strukturierter Zugriff: report_obj.metrics[metric_name][target]
         → bevorzugt, wenn flowcean das Report-Objekt als dict-ähnliche Struktur liefert.
      2. String-Parsing als Fallback: sucht in der String-Repräsentation des Reports
         nach Zeilen, die metric_name und target enthalten, und gibt die letzte Zahl zurück.

    Gibt 0.0 zurück und loggt eine Warnung, wenn die Metrik nicht gefunden wird.
    """
    # Versuch 1: Strukturierter Zugriff via dict-ähnliches Interface.
    try:
        return float(report_obj.metrics[metric_name][target])
    except (AttributeError, KeyError, TypeError):
        pass

    # Versuch 2: String-Parsing als Fallback.
    # Die letzte gültige Zahl in der Trefferzeile ist wahrscheinlich der Metrikwert.
    for line in str(report_obj).splitlines():
        if metric_name in line and target in line:
            candidates = []
            for part in line.split():
                try:
                    candidates.append(float(part))
                except ValueError:
                    continue
            if candidates:
                return candidates[-1]

    logger.warning(
        "Metrik '%s' für Target '%s' nicht gefunden – Report-Inhalt:\n%s",
        metric_name, target, str(report_obj),
    )
    return 0.0


# ─────────────────────────────────────────
#  Plot 1: Füllstände
# ─────────────────────────────────────────

def plot_sensor_data(df: pl.DataFrame, n_tanks: int, topology: str,
                     h_target: List[float], noise_std: float) -> None:
    """
    Visualisiert den zeitlichen Verlauf der Füllstände aller Tanks.

    Erstellt n_tanks übereinanderliegende Subplots (gemeinsame Zeitachse).
    Jeder Subplot zeigt:
      • Blaue Kurve + Füllung: gemessener/simulierter Füllstand.
      • Gestrichelte rote Linie: h_target (Pumpengrenze + Überlaufschutz).

    Speichert das Bild als sensor_plot.png im OUTPUT_DIR.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    t = df["t"].to_numpy()
    topo_label = "Linear (Kette)" if topology == "linear" else "Fully Coupled"
    noise_label = f"  |  Rauschen σ = {noise_std} m" if noise_std > 0 else "  |  kein Rauschen"

    fig, axes = plt.subplots(n_tanks, 1, figsize=(10, 3 * n_tanks), sharex=True)
    if n_tanks == 1:
        axes = [axes]  # Einheitliche Listenbehandlung auch bei einem Tank.

    fig.suptitle(
        f"Füllstände – Topologie: {topo_label}{noise_label}",
        fontsize=13, fontweight="bold"
    )

    for i in range(n_tanks):
        h_vals = df[f"h{i + 1}"].to_numpy()
        c = colors[i % len(colors)]
        axes[i].plot(t, h_vals, color=c, linewidth=1.8)
        axes[i].fill_between(t, h_vals, alpha=0.12, color=c)
        axes[i].axhline(0, color="gray", linewidth=0.8, linestyle="--")
        axes[i].axhline(
            h_target[i], color="red", linewidth=1.2,
            linestyle="--", label=f"h_target = {h_target[i]:.2f} m (Pumpe aus + Obergrenze)"
        )
        axes[i].legend(fontsize=8, loc="upper right")
        axes[i].set_ylabel("Füllstand h [m]", fontsize=9)
        axes[i].set_title(f"Tank {i + 1}", fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Zeit t [s]", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensor_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ sensor_plot.png gespeichert in {OUTPUT_DIR}")


# ─────────────────────────────────────────
#  Plot 2: Modellfehler
# ─────────────────────────────────────────

def plot_learning_results(results: dict, n_tanks: int, model_label: str) -> None:
    """
    Visualisiert MAE und MSE pro Tank als gruppiertes Balkendiagramm.

    Jede Tank-Gruppe zeigt zwei Balken nebeneinander:
      • Blau: MAE (Mean Absolute Error) – mittlerer absoluter Fehler in Metern.
      • Rot:  MSE (Mean Squared Error) – mittlerer quadratischer Fehler.

    Werte werden direkt über den Balken annotiert (6 Nachkommastellen).
    Speichert das Bild als learning_results.png im OUTPUT_DIR.
    """
    x = np.arange(n_tanks)
    width = 0.35
    mae_vals = [results[i]["MAE"] for i in range(n_tanks)]
    mse_vals = [results[i]["MSE"] for i in range(n_tanks)]

    fig, ax = plt.subplots(figsize=(max(6, n_tanks * 2), 5))
    bars1 = ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + width / 2, mse_vals, width, label="MSE", color="#d62728", alpha=0.85)

    # Wert-Annotationen über jedem Balken.
    for bar in (*bars1, *bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.6f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Tank")
    ax.set_ylabel("Fehler")
    ax.set_title(f"Modellfehler pro Tank – {model_label}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tank {i + 1}" for i in range(n_tanks)])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ learning_results.png gespeichert in {OUTPUT_DIR}")


# ─────────────────────────────────────────
#  Plot 3: Vorhersage vs. Realität
# ─────────────────────────────────────────

def plot_predictions(predictions_data: dict, n_tanks: int, model_label: str) -> None:
    """
    Vergleicht vorhergesagte Füllstände mit den tatsächlichen Testwerten.

    Erstellt n_tanks Subplots, jeder zeigt:
      • Farbige Linie: tatsächlicher Füllstand aus den Testdaten.
      • Schwarze gestrichelte Linie: Modellvorhersage.

    Gute Übereinstimmung → Linien liegen nah beieinander.
    Speichert das Bild als predictions_plot.png im OUTPUT_DIR.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    fig, axes = plt.subplots(n_tanks, 1, figsize=(12, 3 * n_tanks), sharex=True)
    if n_tanks == 1:
        axes = [axes]

    fig.suptitle(f"Vorhersage vs. Realität – {model_label}", fontsize=13, fontweight="bold")

    for i in range(n_tanks):
        actual    = predictions_data[i]["actual"]
        predicted = predictions_data[i]["predicted"]
        idx = np.arange(len(actual))
        c = colors[i % len(colors)]

        axes[i].plot(idx, actual, color=c, linewidth=1.5, label="Realität")
        axes[i].plot(idx, predicted, color="black", linewidth=1.0,
                     linestyle="--", alpha=0.75, label="Vorhersage")
        axes[i].set_ylabel("Füllstand h [m]", fontsize=9)
        axes[i].set_title(f"Tank {i + 1}", fontsize=10)
        axes[i].legend(fontsize=8, loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Test-Sample Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ predictions_plot.png gespeichert in {OUTPUT_DIR}")


# ─────────────────────────────────────────
#  Eingabe-Hilfsfunktionen
# ─────────────────────────────────────────

def prompt_choice(prompt: str, valid: tuple) -> str:
    """
    Fordert den Benutzer zur Eingabe einer von mehreren gültigen Optionen auf.
    Wiederholt die Aufforderung, bis eine gültige Option eingegeben wird.
    """
    while True:
        val = input(prompt).strip().lower()
        if val in valid:
            return val
        print(f"  ⚠ Bitte eine der folgenden Optionen eingeben: {valid}")


def prompt_positive_int(prompt: str) -> int:
    """
    Fordert den Benutzer zur Eingabe einer positiven ganzen Zahl auf.
    Wiederholt die Aufforderung bei ungültiger Eingabe.
    """
    while True:
        try:
            val = int(input(prompt).strip())
            if val > 0:
                return val
            print("  ⚠ Wert muss größer als 0 sein.")
        except ValueError:
            print("  ⚠ Bitte eine ganze Zahl eingeben.")


# ─────────────────────────────────────────
#  Hauptprogramm
# ─────────────────────────────────────────

def main() -> None:
    """
    Einstiegspunkt des Programms – koordiniert den gesamten Ablauf:

    1. Konfigurationsphase (interaktiv):
       Topologie, Tankanzahl, Samples, Modell, Pumpen, Ziel-Füllstände,
       Tankflächen, Rauschen, Ventile.

    2. Vorschau-Simulation (20 Schritte, kein Rauschen):
       Gibt einen ersten Eindruck der ODE-Dynamik vor der Hauptsimulation.

    3. Hauptsimulation:
       Sammelt n_samples Datenpunkte mit konfigurierten Einstellungen.
       Erstellt Plot 1 (Füllstandsverläufe).

    4. Feature Engineering:
       SlidingWindow(3) erzeugt aus den Zeitreihen Features der Form
       h{i}_0, h{i}_1 (Vergangenheit) → Ziel h{i}_2 (Zukunft).

    5. Train-Test-Split:
       80 % Training, 20 % Test, keine Zufallsmischung (zeitlicher Verlauf bleibt erhalten).

    6. Inkrementelles Training & Evaluierung (pro Tank):
       Für jeden Tank wird ein separates Modell inkrementell trainiert und
       mit MAE/MSE auf den Testdaten bewertet.

    7. Ergebnisvisualisierung:
       Plot 2 (Modellfehler), Plot 3 (Vorhersage vs. Realität).
    """
    # flowcean-Logging und -Infrastruktur initialisieren.
    flowcean.cli.initialize()

    # ── Schritt 1: Benutzerkonfiguration ──────────────────────────────────────
    topology = prompt_choice(
        "Topologie wählen – 'linear' (Kette) oder 'coupled' (jeder mit jedem): ",
        ("linear", "coupled"),
    )
    n_tanks = prompt_positive_int("Wie viele Tanks sollen simuliert werden? ")
    n_samples = prompt_positive_int(
        "Wie viele Samples sollen gesammelt werden? [Empfehlung: 3000] "
    )
    model_choice = prompt_choice(
        "\nModell wählen – 'hoeffding' (HoeffdingTree) oder 'mlp' (Neuronales Netz): ",
        ("hoeffding", "mlp"),
    )

    model_label = (
        "HoeffdingTreeRegressor" if model_choice == "hoeffding"
        else "MLP Neuronales Netz (river)"
    )
    print(f"→ Modell: {model_label}\n")

    qp_list  = configure_pumps(n_tanks)
    print_pump_summary(qp_list)

    h_target = configure_h_target(n_tanks)
    print_h_target_summary(h_target)

    areas    = configure_tank_areas(n_tanks)
    print_tank_areas_summary(areas)

    noise_std = configure_noise()

    # Ventile: manuell oder alle voll offen (Standard).
    use_custom = input(
        "Ventile manuell konfigurieren? (j/n) [Standard: n, alle voll offen]: "
    ).strip().lower()

    if use_custom == "j":
        valves_between = (
            configure_valves_between_linear(n_tanks)
            if topology == "linear"
            else configure_valves_between_coupled(n_tanks)
        )
        valves_out = configure_valves_out(n_tanks)
    else:
        # Standardkonfiguration: alle Ventile vollständig geöffnet.
        n_between = n_tanks - 1 if topology == "linear" else n_tanks * (n_tanks - 1) // 2
        valves_between = [Valve(open=True, position=1.0) for _ in range(n_between)]
        valves_out     = [Valve(open=True, position=1.0) for _ in range(n_tanks)]

    print_valve_summary(valves_between, valves_out, topology, n_tanks)

    topo_label = "Linear" if topology == "linear" else "Fully Coupled"
    print(f"→ Topologie: {topo_label} | Tanks: {n_tanks} | "
          f"Zwischenventile: {len(valves_between)} | Auslassventile: {n_tanks}\n")

    # ── Schritt 2: Vorschau-Simulation (ohne Rauschen) ─────────────────────────
    # Separates System ohne Rauschen, damit die Vorschau die echte Dynamik zeigt.
    collector_preview = FrameCollector(n_tanks, noise_std=0.0)
    preview_system = build_system(
        topology, n_tanks, valves_between, valves_out, qp_list, h_target, areas
    )
    data_preview = OdeEnvironment(
        preview_system, dt=1.0, map_to_dataframe=collector_preview.collect_frame
    )
    print("Vorschau auf die ersten 20 Schritte:")
    print(collect(data_preview, 20))

    # ── Schritt 3: Hauptsimulation (mit optionalem Rauschen) ───────────────────
    collector = FrameCollector(n_tanks, noise_std=noise_std)
    main_system = build_system(
        topology, n_tanks, valves_between, valves_out, qp_list, h_target, areas
    )
    data_incremental = OdeEnvironment(
        main_system, dt=1.0, map_to_dataframe=collector.collect_frame
    )
    # df_flowcean: flowcean-Wrapper für ML-Pipeline (SlidingWindow, Split etc.)
    # df_plot:     echtes pl.DataFrame für Plots und direkte Indexierung.
    df_flowcean = collect(data_incremental, n_samples)
    df_plot     = collector.concat()

    plot_sensor_data(df_plot, n_tanks, topology, h_target, noise_std)

    # ── Schritt 4: Feature Engineering via SlidingWindow ───────────────────────
    # SlidingWindow(3) erzeugt Tripel: (h_t0, h_t1) → Vorhersage h_t2.
    # Features: h{i}_0, h{i}_1 für jeden Tank i (letzten 2 Zeitschritte).
    # Target:   h{i}_2 (nächster Zeitschritt, 1-Schritt-Prädiktion).
    data   = df_flowcean | SlidingWindow(window_size=3)
    inputs = [f"h{i + 1}_{step}" for i in range(n_tanks) for step in range(2)]
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    # Berechnung der korrekten Indizes für die manuelle Vorhersage-Schleife.
    # SlidingWindow(3) erzeugt (n_samples - 2) Fenster.
    n_windows  = len(df_plot) - 2
    n_train    = int(n_windows * 0.8)
    test_start = n_train  # Ab diesem Index beginnen die Testdaten in df_plot.

    results          = {}  # Speichert MAE/MSE pro Tank-Index.
    predictions_data = {}  # Speichert actual/predicted-Arrays pro Tank-Index.

    # ── Schritte 5–6: Training, Evaluierung & Vorhersage pro Tank ──────────────
    for tank_idx in range(1, n_tanks + 1):
        target_name = f"h{tank_idx}_2"  # Vorhersageziel: Füllstand im nächsten Schritt.
        print(f"\n--- Learning {target_name} ({model_label}) ---")

        # Neues StreamingOfflineEnvironment und neues Modell für jeden Tank
        # (separates Modell pro Ausgang, kein Multi-Output-Lerner).
        train_env = StreamingOfflineEnvironment(train, batch_size=1)
        learner   = build_learner(model_choice)

        # Inkrementelles Training: Modell lernt Sample für Sample.
        t_start       = datetime.now(tz=timezone.utc)
        trained_model = learn_incremental(train_env, learner, inputs, [target_name])
        elapsed_ms    = round((datetime.now(tz=timezone.utc) - t_start).total_seconds() * 1000, 1)
        print(f"Learning {target_name} took {elapsed_ms} ms")

        # Evaluierung auf den Testdaten mit MAE und MSE.
        report = evaluate_offline(
            trained_model, test, inputs, [target_name],
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        print(f"Report for {target_name}:")
        print(report)

        results[tank_idx - 1] = {
            "MAE": extract_metric(report, "MeanAbsoluteError", target_name),
            "MSE": extract_metric(report, "MeanSquaredError", target_name),
        }

        # Vorhersagen manuell generieren (für Plot 3).
        # river_model.predict_one(x) nimmt ein Feature-Dict und gibt einen Skalar zurück.
        river_model  = extract_river_model(trained_model)
        actuals_list = []
        preds_list   = []

        for j in range(test_start, n_windows):
            # Feature-Dict aus den letzten 2 Zeitschritten aufbauen.
            x = {
                f"h{ti + 1}_{step}": float(df_plot[f"h{ti + 1}"][j + step])
                for ti in range(n_tanks)
                for step in range(2)
            }
            pred = river_model.predict_one(x)
            # df_plot direkt verwenden (subscriptable, keine Wrapper-Indirektion).
            actuals_list.append(float(df_plot[f"h{tank_idx}"][j + 2]))
            preds_list.append(float(pred) if pred is not None else 0.0)

        predictions_data[tank_idx - 1] = {
            "actual":    np.array(actuals_list),
            "predicted": np.array(preds_list),
        }

    # ── Schritt 7: Ergebnisvisualisierung ──────────────────────────────────────
    plot_learning_results(results, n_tanks, model_label)
    plot_predictions(predictions_data, n_tanks, model_label)


if __name__ == "__main__":
    main()