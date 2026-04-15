"""
src/twin.py — Digital Twin inference class + NetMob adapter.

Usage — lab / high-fidelity mode (full model):
----------------------------------------------
    from src.twin import DigitalTwin

    twin = DigitalTwin.load("models/manifest.json", variant="usr_safe")
    result = twin.predict({
        "gtpu_kbitss_dn__kbits_tx_s":          400_000,
        "gtpu_kbitss_ngran__gtpu_kbits_tx_s":   10_000,
        "gtpu_packets_dn__packets_tx_delta":     96_000,
        "gtpu_packets_ngran__gtpu_packets_tx_delta": 2_400,
        "user_plane_throughput__l2_3_device_tx_traffic": 410_000,
        "avg_packet_size_bytes":  1250,
        "l2l3_overhead_ratio":    1.86,
    })
    # → {"throughput_gbps": 0.39, "cpu_pct": 1.8, ..., "power_watts": 0.95}

Usage — NetMob / lite mode:
----------------------------
    twin = DigitalTwin.load("models/manifest.json", variant="usr_safe", mode="lite")
    result = twin.predict_from_netmob(
        dl_norm=0.42, ul_norm=0.05,
        c_max_dl_gbps=10.0, c_max_ul_gbps=10.0,
    )
    # → {"throughput_gbps": ..., "cpu_pct": ..., "power_watts": ...}

The twin runs inference in two passes:
    Pass 1 — Layer 1 models predict {throughput, cpu, loss, delay}
    Pass 2 — Layer 2 model predicts power_watts using Pass 1 outputs + inputs
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_pkl(path: str | Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_params() -> dict:
    with open(_PROJECT_ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# NetMob adapter
# ─────────────────────────────────────────────────────────────────────────────

class NetMobAdapter:
    """
    Translates NetMob 2023 normalised traffic loads into the feature space
    expected by the lite digital twin models.

    Parameters
    ----------
    c_max_dl_gbps : float
        Rated downlink capacity of the UPF deployment (Gbps).
    c_max_ul_gbps : float
        Rated uplink capacity of the UPF deployment (Gbps).
    avg_packet_size_bytes : int
        Assumed average packet size for the deployment (bytes).
        Default: 1250 B (matches lab LoadCore configuration).
    interval_sec : int
        Measurement interval in seconds. Default: 3 (matches lab config).
    """

    def __init__(
        self,
        c_max_dl_gbps: float = 10.0,
        c_max_ul_gbps: float = 10.0,
        avg_packet_size_bytes: int = 1250,
        interval_sec: int = 3,
    ):
        self.c_max_dl  = c_max_dl_gbps
        self.c_max_ul  = c_max_ul_gbps
        self.pkt_size  = avg_packet_size_bytes
        self.interval  = interval_sec

    @classmethod
    def from_params(cls, params: dict | None = None) -> "NetMobAdapter":
        """Build adapter from params.yaml adapter section."""
        if params is None:
            params = _load_params()
        a = params.get("adapter", {})
        return cls(
            c_max_dl_gbps         = a.get("c_max_dl_gbps", 10.0),
            c_max_ul_gbps         = a.get("c_max_ul_gbps", 10.0),
            avg_packet_size_bytes = a.get("avg_packet_size_bytes", 1250),
            interval_sec          = a.get("interval_sec", 3),
        )

    def translate(self, dl_norm: float, ul_norm: float) -> dict:
        """
        Convert normalised NetMob loads to model input features.

        Parameters
        ----------
        dl_norm : float
            Normalised downlink load in [0, 1].
        ul_norm : float
            Normalised uplink load in [0, 1].

        Returns
        -------
        dict
            Feature dict compatible with lite model inputs.
        """
        dl_norm = float(np.clip(dl_norm, 0.0, 1.0))
        ul_norm = float(np.clip(ul_norm, 0.0, 1.0))

        dl_kbits_s = dl_norm * self.c_max_dl * 1e6   # Gbps → kbits/s
        ul_kbits_s = ul_norm * self.c_max_ul * 1e6

        # Packet counts derived from throughput
        bits_per_pkt = self.pkt_size * 8
        dl_pkts = (dl_kbits_s * 1e3 * self.interval) / bits_per_pkt
        ul_pkts = (ul_kbits_s * 1e3 * self.interval) / bits_per_pkt

        return {
            "gtpu_kbitss_dn__kbits_tx_s":               dl_kbits_s,
            "gtpu_kbitss_ngran__gtpu_kbits_tx_s":        ul_kbits_s,
            # Extended features for full mode (imputed from config defaults)
            "gtpu_packets_dn__packets_tx_delta":          dl_pkts,
            "gtpu_packets_ngran__gtpu_packets_tx_delta":  ul_pkts,
            "user_plane_throughput__l2_3_device_tx_traffic": dl_kbits_s,
            "avg_packet_size_bytes":  float(self.pkt_size),
            "l2l3_overhead_ratio":    1.86,   # USR deployment default
        }


# ─────────────────────────────────────────────────────────────────────────────
# Digital Twin
# ─────────────────────────────────────────────────────────────────────────────

class DigitalTwin:
    """
    Two-layer digital twin for UPF power and performance prediction.

    Attributes
    ----------
    variant : str
        One of "dpdk", "usr_full", "usr_safe".
    mode : str
        "full" (all offered-load features) or "lite" (throughput TX only).
    """

    L1_OUTPUT_NAMES = {
        "throughput_gbps":   "throughput_gbps",
        "cpu_pct":           "cpu_pct",
        "gtpu_packets_dn__packets_lost_delta": "loss_packets",
        "downlink_one_way_delay_distribution__weighted_mean_delay_us": "delay_us",
    }

    def __init__(
        self,
        variant:   str,
        mode:      Literal["full", "lite"],
        l1_models: dict,   # target → fitted sklearn Pipeline
        l2_model:  object,
        l2_features: list[str],
        l1_targets:  list[str],
        adapter:   NetMobAdapter,
    ):
        self.variant    = variant
        self.mode       = mode
        self._l1        = l1_models
        self._l2        = l2_model
        self._l2_feats  = l2_features
        self._l1_targets = l1_targets
        self.adapter    = adapter

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        manifest_path: str | Path | None = None,
        variant: str = "usr_safe",
        mode: Literal["full", "lite"] = "full",
        params: dict | None = None,
    ) -> "DigitalTwin":
        """
        Load a DigitalTwin from the model manifest produced by train.py.

        Parameters
        ----------
        manifest_path : path to models/manifest.json
        variant       : "dpdk", "usr_full", or "usr_safe"
        mode          : "full" or "lite"
        params        : optional pre-loaded params dict (loads params.yaml if None)
        """
        if manifest_path is None:
            manifest_path = _PROJECT_ROOT / "models" / "manifest.json"
        if params is None:
            params = _load_params()

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        l1_targets = params["layer1_targets"]

        # Load L1 models
        l1_models = {}
        for target in l1_targets:
            key = f"{variant}__layer1__{target}__{mode}"
            if key not in manifest:
                raise KeyError(
                    f"L1 model not found in manifest: {key}\n"
                    f"Run src/train.py first."
                )
            entry = manifest[key]
            l1_models[target] = _load_pkl(_PROJECT_ROOT / entry["path"])

        # Load L2 model
        l2_key = f"{variant}__layer2__power_watts__{mode}"
        if l2_key not in manifest:
            raise KeyError(f"L2 model not found in manifest: {l2_key}")
        l2_entry   = manifest[l2_key]
        l2_model   = _load_pkl(_PROJECT_ROOT / l2_entry["path"])
        l2_features = l2_entry["features"]

        adapter = NetMobAdapter.from_params(params)

        return cls(
            variant=variant,
            mode=mode,
            l1_models=l1_models,
            l2_model=l2_model,
            l2_features=l2_features,
            l1_targets=l1_targets,
            adapter=adapter,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> dict:
        """
        Run the two-layer inference pipeline.

        Parameters
        ----------
        features : dict mapping feature name → scalar value.
            Must include all features expected by the loaded model variant
            (full or lite).  Missing keys are filled with 0.

        Returns
        -------
        dict with keys:
            throughput_gbps, cpu_pct, loss_packets, delay_us, power_watts
        """
        # Pass 1 — Layer 1
        l1_preds = {}
        for target, model in self._l1.items():
            feat_names = model.feature_names_in_ if hasattr(
                model, "feature_names_in_") else None

            if feat_names is None:
                # Pipeline: get from the named step if available
                try:
                    feat_names = model["model"].feature_names_in_
                except Exception:
                    feat_names = list(features.keys())

            row = {f: features.get(f, 0.0) for f in feat_names}
            X   = pd.DataFrame([row])[list(feat_names)]
            l1_preds[target] = float(model.predict(X)[0])

        # Pass 2 — Layer 2
        # Build L2 feature row: offered-load features + l1 predictions
        l2_row = {}
        for feat in self._l2_feats:
            if feat.startswith("l1_pred__"):
                # OOF prediction column from training — fill with L1 output
                target_name = feat[len("l1_pred__"):]
                l2_row[feat] = l1_preds.get(target_name, 0.0)
            else:
                l2_row[feat] = features.get(feat, 0.0)

        X_l2       = pd.DataFrame([l2_row])[self._l2_feats]
        power_pred = float(self._l2.predict(X_l2)[0])

        return {
            "throughput_gbps": l1_preds.get("throughput_gbps", None),
            "cpu_pct":         l1_preds.get("cpu_pct", None),
            "loss_packets":    l1_preds.get(
                "gtpu_packets_dn__packets_lost_delta", None),
            "delay_us":        l1_preds.get(
                "downlink_one_way_delay_distribution__weighted_mean_delay_us",
                None),
            "power_watts":     max(0.0, power_pred),
        }

    def predict_from_netmob(
        self,
        dl_norm: float,
        ul_norm: float,
        c_max_dl_gbps: float | None = None,
        c_max_ul_gbps: float | None = None,
    ) -> dict:
        """
        Convenience wrapper: translate NetMob normalised loads and predict.

        Parameters
        ----------
        dl_norm       : normalised downlink load in [0, 1]
        ul_norm       : normalised uplink load in [0, 1]
        c_max_dl_gbps : override rated DL capacity (Gbps); uses adapter default if None
        c_max_ul_gbps : override rated UL capacity (Gbps); uses adapter default if None

        Returns
        -------
        Same dict as predict(), plus "dl_norm", "ul_norm",
        "dl_gbps_offered", "ul_gbps_offered".
        """
        if c_max_dl_gbps is not None:
            self.adapter.c_max_dl = c_max_dl_gbps
        if c_max_ul_gbps is not None:
            self.adapter.c_max_ul = c_max_ul_gbps

        features = self.adapter.translate(dl_norm, ul_norm)
        result   = self.predict(features)

        result["dl_norm"]         = dl_norm
        result["ul_norm"]         = ul_norm
        result["dl_gbps_offered"] = dl_norm * self.adapter.c_max_dl
        result["ul_gbps_offered"] = ul_norm * self.adapter.c_max_ul
        return result

    def predict_timeseries(
        self,
        dl_norms: list[float],
        ul_norms: list[float],
        c_max_dl_gbps: float | None = None,
        c_max_ul_gbps: float | None = None,
    ) -> pd.DataFrame:
        """
        Predict over a time series of normalised loads.

        Parameters
        ----------
        dl_norms : list of normalised DL loads (one per timestep)
        ul_norms : list of normalised UL loads (one per timestep)

        Returns
        -------
        pd.DataFrame with one row per timestep and all prediction columns.
        """
        rows = [
            self.predict_from_netmob(dl, ul, c_max_dl_gbps, c_max_ul_gbps)
            for dl, ul in zip(dl_norms, ul_norms)
        ]
        return pd.DataFrame(rows)

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DigitalTwin(variant={self.variant!r}, mode={self.mode!r}, "
            f"l1_targets={self._l1_targets})"
        )
