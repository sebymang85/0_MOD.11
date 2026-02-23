import os

# =============================================================================
# Limita threading interno di NumPy/OpenBLAS/MKL per evitare oversubscription
# (DEVE stare prima di importare librerie che tirano dentro numpy)
# =============================================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import itertools
import time
from datetime import datetime
from functools import reduce
from operator import mul
from multiprocessing import Pool, cpu_count
from dotenv import load_dotenv

import numpy as np
import pandas as pd

import simulator  # il tuo simulator.py


# -------------------------------
# CONFIGURAZIONE PARAMETRI
# -------------------------------

PARAMS = [
    ("VOLUME_LOOKBACK", int),
    ("VOLUME_THRESHOLD", float),
    ("VOLUME_THRESHOLD_MAX", float),

    ("PRICE_INCREASE_WINDOW", int),
    ("MIN_PRICE_INCREASE_PERCENT", float),

    ("MIN_CURRENT_VOLUME", float),
    ("MAX_CURRENT_VOLUME", float),

    ("SPIKE_COOLDOWN_CANDLES", int),

    ("UTC_HOUR_MIN", int),
    ("UTC_HOUR_MAX", int),

    ("UTC_WEEKDAY", str),
    ("PRICE_RATIO_AT_SPIKE_MODE", str),

    ("PRICE_RATIO_AT_SPIKE_MIN", float),
    ("PRICE_RATIO_AT_SPIKE_MAX", float),

    ("HL_MIN_PCT", float),
    ("BODY_RATIO_MIN", float),
]

ENV_TO_CONFIG_KEY = {
    "VOLUME_LOOKBACK": "volume_lookback",
    "VOLUME_THRESHOLD": "volume_threshold",
    "VOLUME_THRESHOLD_MAX": "volume_threshold_max",

    "PRICE_INCREASE_WINDOW": "price_increase_window",
    "MIN_PRICE_INCREASE_PERCENT": "min_price_increase_percent",

    "MIN_CURRENT_VOLUME": "min_current_volume",
    "MAX_CURRENT_VOLUME": "max_current_volume",

    "SPIKE_COOLDOWN_CANDLES": "spike_cooldown_candles",

    "UTC_HOUR_MIN": "utc_hour_min",
    "UTC_HOUR_MAX": "utc_hour_max",

    "UTC_WEEKDAY": "utc_weekday",
    "PRICE_RATIO_AT_SPIKE_MODE": "price_ratio_at_spike_mode",

    "PRICE_RATIO_AT_SPIKE_MIN": "price_ratio_at_spike_min",
    "PRICE_RATIO_AT_SPIKE_MAX": "price_ratio_at_spike_max",

    "HL_MIN_PCT": "hl_min_pct",
    "BODY_RATIO_MIN": "body_ratio_min",
}

# =============================================================================
# SOFT_PARAMS = filtri post-qualifica (NON cambiano la scansione core)
# => UTC_HOUR_* DEVE stare qui per avere il boost vero.
# =============================================================================
SOFT_PARAMS = {
    "VOLUME_THRESHOLD_MAX",
    "MIN_CURRENT_VOLUME",
    "MAX_CURRENT_VOLUME",
    "UTC_HOUR_MIN",
    "UTC_HOUR_MAX",
    "UTC_WEEKDAY",
    "PRICE_RATIO_AT_SPIKE_MODE",
    "PRICE_RATIO_AT_SPIKE_MIN",
    "PRICE_RATIO_AT_SPIKE_MAX",
    "HL_MIN_PCT",
    "BODY_RATIO_MIN",
}

# HARD = cambia il set di candidati/TP/SL/cooldown etc.
HARD_PARAMS = [name for (name, _ptype) in PARAMS if name not in SOFT_PARAMS]


def load_optimizer_config(env_path: str = "optimizer.env"):
    """
    Se <NAME>_ENABLED non esiste -> enabled=0 (compatibilità).
    Se enabled=1: richiede MIN/MAX/STEP (solo numerici).
    """
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        raise FileNotFoundError(f"File {env_path} non trovato")

    range_config = {}
    enabled_flags = {}

    for name, ptype in PARAMS:
        enabled_key = f"{name}_ENABLED"

        enabled_str = os.getenv(enabled_key, None)
        if enabled_str is None:
            enabled = False
        else:
            enabled = str(enabled_str).strip() == "1"

        enabled_flags[name] = enabled

        if not enabled:
            continue

        if ptype is str:
            raise ValueError(
                f"{enabled_key}=1 ma {name} è stringa (enum/list). "
                "Questo optimizer supporta solo range numerici MIN/MAX/STEP. "
                "Mettilo ENABLED=0 e gestiscilo da simulator.env."
            )

        min_key = f"{name}_MIN"
        max_key = f"{name}_MAX"
        step_key = f"{name}_STEP"

        try:
            min_val = float(os.environ[min_key])
            max_val = float(os.environ[max_key])
            step_val = float(os.environ[step_key])
        except KeyError as e:
            raise KeyError(
                f"Variabile {e.args[0]} mancante in {env_path} per il parametro {name}"
            )

        if step_val <= 0:
            raise ValueError(f"{step_key} deve essere > 0 (parametro {name})")

        if max_val < min_val:
            raise ValueError(
                f"{max_key} ({max_val}) deve essere >= {min_key} ({min_val}) per {name}"
            )

        range_config[name] = {"min": min_val, "max": max_val, "step": step_val}

    output_csv = os.environ.get("OPTIMIZER_OUTPUT_CSV", "optimizer_results.csv")
    return range_config, output_csv, enabled_flags


def generate_values(min_val: float, max_val: float, step: float, is_int: bool):
    """Genera valori MIN..MAX inclusi (aggiunge MAX se serve)."""
    n_steps_float = (max_val - min_val) / step
    n_steps = int(n_steps_float)

    values = []
    for i in range(n_steps + 1):
        v = min_val + step * i
        values.append(v)

    if values:
        if max_val - values[-1] > 1e-9:
            values.append(max_val)
    else:
        values = [min_val]

    if is_int:
        return [int(round(v)) for v in values]
    return values


def _stringify_utc_weekday(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        try:
            vals = [str(int(x)) for x in sorted(list(value))]
            return ",".join(vals)
        except Exception:
            return ",".join([str(x) for x in value])
    return str(value).strip()


def format_float_for_eu(value, decimals: int) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}".replace(".", ",")
    except (ValueError, TypeError):
        return ""


# =============================================================================
# SPIKES FILE MINIMO (per backtest): symbol;timestamp;datetime
# =============================================================================
_SPIKES_MIN_COLUMNS = ["symbol", "timestamp", "datetime"]


def _write_spikes_min_csv_from_arrays(symbol_arr, ts_arr, dt_str_arr, idx_arr, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8", newline="", buffering=1024 * 1024) as f:
        writer = csv.writer(
            f,
            delimiter=";",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writerow(_SPIKES_MIN_COLUMNS)

        if idx_arr.size == 0:
            return

        # write rows
        for i in idx_arr:
            writer.writerow([symbol_arr[i], str(int(ts_arr[i])), dt_str_arr[i]])


# =============================================================================
# STREAMING optimizer_results.csv
# =============================================================================
_OPT_FIELDNAMES = [
    "VOLUME_LOOKBACK",
    "VOLUME_THRESHOLD",
    "VOLUME_THRESHOLD_MAX",
    "PRICE_INCREASE_WINDOW",
    "MIN_PRICE_INCREASE_PERCENT",
    "MIN_CURRENT_VOLUME",
    "MAX_CURRENT_VOLUME",
    "SPIKE_COOLDOWN_CANDLES",
    "UTC_HOUR_MIN",
    "UTC_HOUR_MAX",
    "UTC_WEEKDAY",
    "PRICE_RATIO_AT_SPIKE_MODE",
    "PRICE_RATIO_AT_SPIKE_MIN",
    "PRICE_RATIO_AT_SPIKE_MAX",
    "HL_MIN_PCT",
    "BODY_RATIO_MIN",
    "SPIKES_FILE",
    "efficiency_long",
    "avg_price_increase_long",
    "avg_lead_time_long",
    "efficiency_short",
    "avg_price_decrease_short",
    "avg_lead_time_short",
    "total_spikes",
]


def _write_optimizer_header(writer: csv.writer) -> None:
    writer.writerow(_OPT_FIELDNAMES)


def _write_optimizer_row(writer: csv.writer, formatted_row: dict) -> None:
    writer.writerow([formatted_row.get(col, "") for col in _OPT_FIELDNAMES])


def _format_optimizer_row_for_csv(row: dict) -> dict:
    """
    Replica formattazione EU come prima:
      - int -> stringa
      - float param -> 2 decimali EU
      - efficienze -> 4 decimali EU
      - avg_price_* -> 6 decimali EU
      - avg_lead_time_* -> 2 decimali EU
      - None -> ""
    """
    out = dict(row) if row else {}

    for col in ["VOLUME_LOOKBACK", "PRICE_INCREASE_WINDOW", "SPIKE_COOLDOWN_CANDLES", "total_spikes"]:
        v = out.get(col)
        out[col] = str(int(v)) if v not in ("", None) else ""

    for col in ["UTC_HOUR_MIN", "UTC_HOUR_MAX"]:
        v = out.get(col)
        out[col] = str(int(v)) if v not in ("", None) else ""

    for col in [
        "VOLUME_THRESHOLD",
        "VOLUME_THRESHOLD_MAX",
        "MIN_PRICE_INCREASE_PERCENT",
        "MIN_CURRENT_VOLUME",
        "MAX_CURRENT_VOLUME",
        "PRICE_RATIO_AT_SPIKE_MIN",
        "PRICE_RATIO_AT_SPIKE_MAX",
    ]:
        out[col] = format_float_for_eu(out.get(col), 2)

    for col in ["efficiency_long", "efficiency_short"]:
        out[col] = format_float_for_eu(out.get(col), 4)

    for col in ["avg_price_increase_long", "avg_price_decrease_short"]:
        out[col] = format_float_for_eu(out.get(col), 6)

    for col in ["avg_lead_time_long", "avg_lead_time_short"]:
        out[col] = format_float_for_eu(out.get(col), 2)

    for k in _OPT_FIELDNAMES:
        if k not in out or out[k] is None:
            out[k] = ""

    return out


# =============================================================================
# WORKER GLOBALS
# =============================================================================
_worker_simulator = None

_SOFT_PARAM_NAMES = []
_SOFT_COMBOS = []
_SOFT_TOTAL = 1
_PROFILE_ENABLED = False
_SPIKES_BATCH_SIZE = 1000


def _dt_string_from_ts_cached(ts_ms: int, cache: dict) -> str:
    key = int(ts_ms)
    out = cache.get(key)
    if out is not None:
        return out
    out = datetime.utcfromtimestamp(key / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
    if len(cache) >= 200000:
        cache.clear()
    cache[key] = out
    return out


def _write_spikes_batch(arr: dict, pending_writes: list, dt_cache: dict) -> None:
    if not pending_writes:
        return

    symbol_arr = arr["symbol"]
    ts_arr = arr["ts"]
    dt_arr = arr.get("dt_str")

    for spikes_file, idx_sel in pending_writes:
        with open(spikes_file, "w", encoding="utf-8", newline="", buffering=1024 * 1024) as f:
            writer = csv.writer(
                f,
                delimiter=";",
                quoting=csv.QUOTE_MINIMAL,
                lineterminator="\n",
            )
            writer.writerow(_SPIKES_MIN_COLUMNS)

            if idx_sel.size == 0:
                continue

            for i in idx_sel:
                i_int = int(i)
                ts_i = int(ts_arr[i_int])

                dt_s = ""
                if dt_arr is not None and dt_arr.size > i_int:
                    try:
                        dt_s = dt_arr[i_int]
                    except Exception:
                        dt_s = ""
                if not dt_s:
                    dt_s = _dt_string_from_ts_cached(ts_i, dt_cache)

                writer.writerow([symbol_arr[i_int], str(ts_i), dt_s])

    pending_writes.clear()


def _parse_weekday_to_set(value):
    """
    Normalizza UTC_WEEKDAY in set di int oppure None.
    """
    if value is None:
        return None
    if isinstance(value, set):
        return set(int(x) for x in value)
    if isinstance(value, (list, tuple)):
        return set(int(x) for x in value)
    s = str(value).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        return None
    return set(int(p) for p in parts)


def _init_worker(base_config: dict, soft_param_names: list, soft_combos: list, soft_total: int, profile_enabled: bool):
    """
    Init worker:
    - logging
    - TradingSimulator sequenziale
    - carica tutti i simboli in RAM
    - cache np_data per DF cached
    """
    global _worker_simulator, _SOFT_PARAM_NAMES, _SOFT_COMBOS, _SOFT_TOTAL, _PROFILE_ENABLED
    global _SPIKES_BATCH_SIZE

    _SOFT_PARAM_NAMES = list(soft_param_names) if soft_param_names else []
    _SOFT_COMBOS = list(soft_combos) if soft_combos else [()]
    _SOFT_TOTAL = int(soft_total) if soft_total and int(soft_total) > 0 else 1
    _PROFILE_ENABLED = bool(profile_enabled)
    try:
        _SPIKES_BATCH_SIZE = max(1, int(os.environ.get("OPTIMIZER_SPIKES_BATCH_SIZE", "1000")))
    except ValueError:
        _SPIKES_BATCH_SIZE = 1000

    cfg = dict(base_config)

    log_level = str(cfg.get("log_level", "ERROR"))
    simulator.setup_detailed_logging(log_level, log_to_file=False)

    cfg["max_workers"] = 1
    cfg["force_sequential"] = True
    cfg["save_detailed_csv"] = False
    cfg["save_us_csv"] = False
    cfg["core_subprofile"] = bool(profile_enabled)
    cfg["core_include_dt_strings"] = False

    sim = simulator.TradingSimulator(cfg)

    symbols = sim.get_available_symbols()
    for sym in symbols:
        sim.load_symbol_data(sym)

    orig_prepare = sim._prepare_symbol_numpy
    _np_cache = {}

    def _cached_prepare(df):
        key = id(df)
        cached = _np_cache.get(key)
        if cached is not None:
            return cached
        out = orig_prepare(df)
        _np_cache[key] = out
        return out

    sim._prepare_symbol_numpy = _cached_prepare

    # C2 minimale: pre-warm cache numpy per tutti i simboli gia' in RAM nel worker.
    # Evita chiamate ripetute al path load/prepare durante la core scan.
    for sym in symbols:
        try:
            sim.load_symbol_numpy_data(sym)
        except Exception:
            # fallback sicuro: il path standard restera' disponibile a runtime
            continue

    _worker_simulator = sim


def _neutralize_post_filters_for_core_scan(sim) -> dict:
    """
    Disattiva filtri post-qualifica prima della core scan (riusabile per SOFT).
    """
    saved = {
        "volume_threshold_max": getattr(sim, "volume_threshold_max", None),
        "min_current_volume": getattr(sim, "min_current_volume", 0.0),
        "max_current_volume": getattr(sim, "max_current_volume", None),
        "utc_hour_min": getattr(sim, "utc_hour_min", None),
        "utc_hour_max": getattr(sim, "utc_hour_max", None),
        "utc_weekday_set": getattr(sim, "utc_weekday_set", None),
        "config_utc_weekday": sim.config.get("utc_weekday"),
        "hl_min_pct": getattr(sim, "hl_min_pct", None),
        "body_ratio_min": getattr(sim, "body_ratio_min", None),
    }

    sim.volume_threshold_max = None
    sim.min_current_volume = 0.0
    sim.max_current_volume = None
    sim.utc_hour_min = None
    sim.utc_hour_max = None
    sim.utc_weekday_set = None
    sim.hl_min_pct = None
    sim.body_ratio_min = None

    sim.config["volume_threshold_max"] = None
    sim.config["min_current_volume"] = 0.0
    sim.config["max_current_volume"] = None
    sim.config["utc_hour_min"] = None
    sim.config["utc_hour_max"] = None
    sim.config["utc_weekday"] = None
    sim.config["hl_min_pct"] = None
    sim.config["body_ratio_min"] = None

    return saved


def _restore_post_filters_after_core_scan(sim, saved: dict) -> None:
    sim.volume_threshold_max = saved.get("volume_threshold_max")
    sim.min_current_volume = saved.get("min_current_volume")
    sim.max_current_volume = saved.get("max_current_volume")
    sim.utc_hour_min = saved.get("utc_hour_min")
    sim.utc_hour_max = saved.get("utc_hour_max")
    sim.utc_weekday_set = saved.get("utc_weekday_set")
    sim.hl_min_pct = saved.get("hl_min_pct")
    sim.body_ratio_min = saved.get("body_ratio_min")

    sim.config["volume_threshold_max"] = saved.get("volume_threshold_max")
    sim.config["min_current_volume"] = saved.get("min_current_volume")
    sim.config["max_current_volume"] = saved.get("max_current_volume")
    sim.config["utc_hour_min"] = saved.get("utc_hour_min")
    sim.config["utc_hour_max"] = saved.get("utc_hour_max")
    sim.config["utc_weekday"] = saved.get("config_utc_weekday")
    sim.config["hl_min_pct"] = saved.get("hl_min_pct")
    sim.config["body_ratio_min"] = saved.get("body_ratio_min")


def _build_core_arrays(core_spikes):
    """
    Precompute array per HARD:
    - evita dict access nel loop SOFT
    - permette mask numpy veloce

    Supporta:
      - core_spikes = list[dict] (formato attuale)
      - core_spikes = dict di arrays (formato "compatto" per Livello B)
    """
    # --- formato compatto (Livello B) ---
    if isinstance(core_spikes, dict) and ("ts" in core_spikes or "timestamp" in core_spikes):
        ts_arr = core_spikes.get("ts", None)
        if ts_arr is None:
            ts_arr = core_spikes.get("timestamp", None)

        if ts_arr is None:
            # fallback, trattalo come vuoto
            n = 0
        else:
            ts_arr = np.asarray(ts_arr, dtype=np.int64)
            n = int(ts_arr.shape[0])

        if n == 0:
            return {
                "n": 0,
                "symbol": np.array([], dtype=object),
                "ts": np.array([], dtype=np.int64),
                "dt_str": np.array([], dtype=object),
                "curr_vol": np.array([], dtype=np.float64),
                "vol_ratio": np.array([], dtype=np.float64),
                "price_ratio": np.array([], dtype=np.float64),
                "hl_pct": np.array([], dtype=np.float64),
                "body_ratio": np.array([], dtype=np.float64),
                "is_ok": np.array([], dtype=bool),
                "is_ok_short": np.array([], dtype=bool),
                "price_increase": np.array([], dtype=np.float64),
                "lead_time": np.array([], dtype=np.int32),
                "price_decrease": np.array([], dtype=np.float64),
                "lead_time_down": np.array([], dtype=np.int32),
                "utc_hour": np.array([], dtype=np.int16),
                "utc_weekday": np.array([], dtype=np.int8),
            }

        symbol = np.asarray(core_spikes.get("symbol", [""] * n), dtype=object)
        dt_str = np.asarray(core_spikes.get("dt_str", core_spikes.get("datetime", [""] * n)), dtype=object)

        curr_vol = np.asarray(core_spikes.get("curr_vol", core_spikes.get("current_volume", np.zeros(n))), dtype=np.float64)
        vol_ratio = np.asarray(core_spikes.get("vol_ratio", core_spikes.get("volume_ratio", np.zeros(n))), dtype=np.float64)
        price_ratio = np.asarray(core_spikes.get("price_ratio", np.zeros(n)), dtype=np.float64)
        hl_pct = np.asarray(core_spikes.get("hl_pct", np.zeros(n)), dtype=np.float64)
        body_ratio = np.asarray(core_spikes.get("body_ratio", np.zeros(n)), dtype=np.float64)

        is_ok = np.asarray(core_spikes.get("is_ok", np.zeros(n, dtype=bool)), dtype=bool)
        is_ok_short = np.asarray(core_spikes.get("is_ok_short", np.zeros(n, dtype=bool)), dtype=bool)

        price_increase = np.asarray(core_spikes.get("price_increase", np.zeros(n)), dtype=np.float64)
        lead_time = np.asarray(core_spikes.get("lead_time", np.zeros(n)), dtype=np.int32)

        price_decrease = np.asarray(core_spikes.get("price_decrease", np.zeros(n)), dtype=np.float64)
        lead_time_down = np.asarray(core_spikes.get("lead_time_down", np.zeros(n)), dtype=np.int32)

        ts = ts_arr

        # UTC hour con aritmetica epoch (super veloce)
        utc_hour = ((ts // 3600000) % 24).astype(np.int16)

        # weekday: aritmetica epoch (no pandas)
        # 1970-01-01 = giovedì => weekday=3 se Monday=0
        utc_weekday = ((ts // 86400000 + 3) % 7).astype(np.int8)

        return {
            "n": n,
            "symbol": symbol,
            "ts": ts,
            "dt_str": dt_str,
            "curr_vol": curr_vol,
            "vol_ratio": vol_ratio,
            "price_ratio": price_ratio,
            "hl_pct": hl_pct,
            "body_ratio": body_ratio,
            "is_ok": is_ok,
            "is_ok_short": is_ok_short,
            "price_increase": price_increase,
            "lead_time": lead_time,
            "price_decrease": price_decrease,
            "lead_time_down": lead_time_down,
            "utc_hour": utc_hour,
            "utc_weekday": utc_weekday,
        }

    # --- formato attuale list[dict] ---
    n = len(core_spikes)
    if n == 0:
        return {
            "n": 0,
            "symbol": np.array([], dtype=object),
            "ts": np.array([], dtype=np.int64),
            "dt_str": np.array([], dtype=object),
            "curr_vol": np.array([], dtype=np.float64),
            "vol_ratio": np.array([], dtype=np.float64),
            "price_ratio": np.array([], dtype=np.float64),
            "hl_pct": np.array([], dtype=np.float64),
            "body_ratio": np.array([], dtype=np.float64),
            "is_ok": np.array([], dtype=bool),
            "is_ok_short": np.array([], dtype=bool),
            "price_increase": np.array([], dtype=np.float64),
            "lead_time": np.array([], dtype=np.int32),
            "price_decrease": np.array([], dtype=np.float64),
            "lead_time_down": np.array([], dtype=np.int32),
            "utc_hour": np.array([], dtype=np.int16),
            "utc_weekday": np.array([], dtype=np.int8),
        }

    symbol = np.empty(n, dtype=object)
    ts = np.empty(n, dtype=np.int64)
    dt_str = np.empty(n, dtype=object)

    curr_vol = np.empty(n, dtype=np.float64)
    vol_ratio = np.empty(n, dtype=np.float64)
    price_ratio = np.empty(n, dtype=np.float64)
    hl_pct = np.empty(n, dtype=np.float64)
    body_ratio = np.empty(n, dtype=np.float64)

    is_ok = np.empty(n, dtype=bool)
    is_ok_short = np.empty(n, dtype=bool)

    price_increase = np.empty(n, dtype=np.float64)
    lead_time = np.empty(n, dtype=np.int32)

    price_decrease = np.empty(n, dtype=np.float64)
    lead_time_down = np.empty(n, dtype=np.int32)

    for i, s in enumerate(core_spikes):
        symbol[i] = s.get("symbol", "")
        tsv = s.get("timestamp", 0)
        try:
            ts[i] = int(tsv)
        except Exception:
            ts[i] = 0

        d = s.get("datetime")
        dt_str[i] = "" if d is None else str(d)

        curr_vol[i] = float(s.get("current_volume", 0.0))
        vol_ratio[i] = float(s.get("volume_ratio", 0.0))
        price_ratio[i] = float(s.get("price_ratio", 0.0))
        hl_pct[i] = float(s.get("hl_pct", 0.0))
        body_ratio[i] = float(s.get("body_ratio", 0.0))

        is_ok[i] = bool(s.get("is_ok", False))
        is_ok_short[i] = bool(s.get("is_ok_short", False))

        price_increase[i] = float(s.get("price_increase", 0.0))
        lead_time[i] = int(s.get("lead_time", 0) or 0)

        price_decrease[i] = float(s.get("price_decrease", 0.0))
        lead_time_down[i] = int(s.get("lead_time_down", 0) or 0)

    # UTC hour con aritmetica epoch (super veloce)
    utc_hour = ((ts // 3600000) % 24).astype(np.int16)

    # weekday: aritmetica epoch (no pandas)
    # 1970-01-01 = giovedì => weekday=3 se Monday=0
    utc_weekday = ((ts // 86400000 + 3) % 7).astype(np.int8)

    return {
        "n": n,
        "symbol": symbol,
        "ts": ts,
        "dt_str": dt_str,
        "curr_vol": curr_vol,
        "vol_ratio": vol_ratio,
        "price_ratio": price_ratio,
        "hl_pct": hl_pct,
        "body_ratio": body_ratio,
        "is_ok": is_ok,
        "is_ok_short": is_ok_short,
        "price_increase": price_increase,
        "lead_time": lead_time,
        "price_decrease": price_decrease,
        "lead_time_down": lead_time_down,
        "utc_hour": utc_hour,
        "utc_weekday": utc_weekday,
    }


def _utc_hour_mask(utc_hour_arr, hmin: int, hmax: int):
    """
    Semantica [MIN, MAX) con supporto wrap mezzanotte.
    """
    if hmin < hmax:
        return (utc_hour_arr >= hmin) & (utc_hour_arr < hmax)
    return (utc_hour_arr >= hmin) | (utc_hour_arr < hmax)


def _price_ratio_mask(price_ratio_arr, mode: str, min_v, max_v):
    """
    MODE=TAILS:
      - MIN -> pr <= MIN
      - MAX -> pr >= MAX
      - entrambi -> OR
    MODE=RANGE:
      - MIN <= pr <= MAX
    """
    m = str(mode).strip().upper() if mode is not None else "TAILS"

    if m == "RANGE":
        if min_v is None or max_v is None:
            return np.zeros(price_ratio_arr.shape[0], dtype=bool)
        mn = float(min_v)
        mx = float(max_v)
        return (price_ratio_arr >= mn) & (price_ratio_arr <= mx)

    # TAILS
    if min_v is None and max_v is None:
        return np.ones(price_ratio_arr.shape[0], dtype=bool)

    keep = np.zeros(price_ratio_arr.shape[0], dtype=bool)
    if min_v is not None:
        keep |= (price_ratio_arr <= float(min_v))
    if max_v is not None:
        keep |= (price_ratio_arr >= float(max_v))
    return keep


def _compute_metrics_for_optimizer(arr, idx_arr):
    """
    Calcolo metriche FAST su idx selezionati.

    Ottimizzazione: evita slicing ripetuti arr[key][idx_arr] nel path caldo SOFT.
    """
    total = int(idx_arr.size)
    if total == 0:
        return {
            "total_spikes": 0,
            "efficiency_long": 0.0,
            "avg_price_increase_long": 0.0,
            "avg_lead_time_long": 0.0,
            "efficiency_short": 0.0,
            "avg_price_decrease_short": 0.0,
            "avg_lead_time_short": 0.0,
        }

    sel_is_ok = arr["is_ok"][idx_arr]
    sel_is_ok_short = arr["is_ok_short"][idx_arr]
    sel_price_increase = arr["price_increase"][idx_arr]
    sel_lead_time = arr["lead_time"][idx_arr]
    sel_price_decrease = arr["price_decrease"][idx_arr]
    sel_lead_time_down = arr["lead_time_down"][idx_arr]

    ok_long_count = int(np.count_nonzero(sel_is_ok))
    ok_short_count = int(np.count_nonzero(sel_is_ok_short))

    efficiency_long = ok_long_count / total if total > 0 else 0.0
    efficiency_short = ok_short_count / total if total > 0 else 0.0

    if ok_long_count > 0:
        pi_ok = sel_price_increase[sel_is_ok]
        avg_price_increase_long = float(np.mean(pi_ok)) if pi_ok.size else 0.0

        lt_ok = sel_lead_time[sel_is_ok]
        lt_ok = lt_ok[lt_ok > 0]
        avg_lead_time_long = float(np.mean(lt_ok)) if lt_ok.size else 0.0
    else:
        avg_price_increase_long = 0.0
        avg_lead_time_long = 0.0

    if ok_short_count > 0:
        pd_ok = sel_price_decrease[sel_is_ok_short]
        avg_price_decrease_short = float(np.mean(pd_ok)) if pd_ok.size else 0.0

        ltd_ok = sel_lead_time_down[sel_is_ok_short]
        ltd_ok = ltd_ok[ltd_ok > 0]
        avg_lead_time_short = float(np.mean(ltd_ok)) if ltd_ok.size else 0.0
    else:
        avg_price_decrease_short = 0.0
        avg_lead_time_short = 0.0

    return {
        "total_spikes": total,
        "efficiency_long": float(efficiency_long),
        "avg_price_increase_long": float(avg_price_increase_long),
        "avg_lead_time_long": float(avg_lead_time_long),
        "efficiency_short": float(efficiency_short),
        "avg_price_decrease_short": float(avg_price_decrease_short),
        "avg_lead_time_short": float(avg_lead_time_short),
    }



def _chunked(iterable, chunk_size: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf




def _take_next_chunk(iterator, chunk_size: int):
    out = []
    for _ in range(max(1, int(chunk_size))):
        try:
            out.append(next(iterator))
        except StopIteration:
            break
    return out


def _auto_chunk_size(num_hard_groups: int, num_workers: int, chunk_size_env: str) -> int:
    raw = (chunk_size_env or '').strip().lower()
    if raw and raw != 'auto':
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    # AUTO: crea ~16 task per worker (tradeoff tra overhead IPC e bilanciamento)
    workers = max(1, int(num_workers))
    target_tasks = max(workers * 16, 1)
    return max(1, int(np.ceil(num_hard_groups / target_tasks)))


def _adaptive_next_chunk_size(current_size: int, elapsed_s: float, target_s: float = 2.0) -> int:
    size = max(1, int(current_size))
    if elapsed_s <= 0:
        return size
    if elapsed_s < (target_s * 0.7):
        return min(size * 2, 512)
    if elapsed_s > (target_s * 1.6):
        return max(size // 2, 1)
    return size


def _run_simulation_chunk(chunk):
    """
    Per ogni gruppo HARD:
      - applica HARD params
      - core scan UNA volta (filtri post neutralizzati)
      - costruisce arrays una volta
      - per ogni combinazione SOFT:
          - mask numpy
          - scrive spikes file (nome stabile)
          - calcola metriche veloci (identiche a quelle che ti servono nel CSV)
    """
    global _worker_simulator, _SOFT_PARAM_NAMES, _SOFT_COMBOS, _SOFT_TOTAL, _PROFILE_ENABLED
    results = []
    stats = {
        "hard_groups": 0,
        "core_scan_s": 0.0,
        "array_build_s": 0.0,
        "soft_eval_s": 0.0,
        "csv_write_s": 0.0,
        "rows": 0,
        "core_load_np_s": 0.0,
        "core_extrema_s": 0.0,
        "core_candidate_build_s": 0.0,
        "core_candidate_loop_s": 0.0,
        "core_post_filter_s": 0.0,
        "core_price_prep_s": 0.0,
        "core_price_eval_s": 0.0,
        "core_datetime_s": 0.0,
    }

    os.makedirs("simulation_results", exist_ok=True)
    sim = _worker_simulator
    spikes_batch = []
    dt_cache = {}

    def _kfloat(x):
        if x is None:
            return None
        try:
            return round(float(x), 12)
        except Exception:
            return None

    for hard_group in chunk:
        if not hard_group:
            continue

        hard_index, hard_full_params = hard_group

        # Applica HARD params
        for env_name, config_key in ENV_TO_CONFIG_KEY.items():
            if env_name in hard_full_params:
                value = hard_full_params[env_name]
                sim.config[config_key] = value
                setattr(sim, config_key, value)

        # Core scan riusabile: disattiva filtri post prima della scansione
        t_core = time.perf_counter()
        saved_filters = _neutralize_post_filters_for_core_scan(sim)
        core_spikes = sim.run_simulation_core_spikes()
        _restore_post_filters_after_core_scan(sim, saved_filters)
        stats["core_scan_s"] += (time.perf_counter() - t_core)
        if _PROFILE_ENABLED and isinstance(core_spikes, dict):
            cprof = core_spikes.get("__core_profile", {})
            stats["core_load_np_s"] += float(cprof.get("load_np_s", 0.0))
            stats["core_extrema_s"] += float(cprof.get("extrema_s", 0.0))
            stats["core_candidate_build_s"] += float(cprof.get("candidate_build_s", 0.0))
            stats["core_candidate_loop_s"] += float(cprof.get("candidate_loop_s", 0.0))
            stats["core_post_filter_s"] += float(cprof.get("post_filter_s", 0.0))
            stats["core_price_prep_s"] += float(cprof.get("price_prep_s", 0.0))
            stats["core_price_eval_s"] += float(cprof.get("price_eval_s", 0.0))
            stats["core_datetime_s"] += float(cprof.get("datetime_s", 0.0))

        # Precompute arrays per HARD
        t_arr = time.perf_counter()
        arr = _build_core_arrays(core_spikes)
        stats["array_build_s"] += (time.perf_counter() - t_arr)
        stats["hard_groups"] += 1
        n = arr["n"]
        if n == 0:
            pass

        base_all_idx = np.arange(n, dtype=np.int64) if n > 0 else np.array([], dtype=np.int64)

        # cache per HARD group (maschere + parse valori SOFT)
        mask_cache = {}
        weekday_cache = {}
        hour_cache = {}

        def _get_mask(key, builder):
            m = mask_cache.get(key)
            if m is not None:
                return m
            m = builder()
            mask_cache[key] = m
            return m

        for soft_idx, soft_values in enumerate(_SOFT_COMBOS):
            combination_index = (hard_index * _SOFT_TOTAL) + soft_idx + 1

            # full param values = hard + soft
            full_param_values = dict(hard_full_params)
            if _SOFT_PARAM_NAMES:
                for name, value in zip(_SOFT_PARAM_NAMES, soft_values):
                    full_param_values[name] = value

            # SANITY: UTC_HOUR_MIN == UTC_HOUR_MAX -> invalid (policy simulator)
            if "UTC_HOUR_MIN" in full_param_values and "UTC_HOUR_MAX" in full_param_values:
                try:
                    if int(full_param_values["UTC_HOUR_MIN"]) == int(full_param_values["UTC_HOUR_MAX"]):
                        continue
                except Exception:
                    continue

            # ==== Costruisci mask (numpy) ====
            t_soft = time.perf_counter()
            if n == 0:
                idx_sel = base_all_idx
            else:
                mask = np.ones(n, dtype=bool)

                # min/max current volume
                min_cv = full_param_values.get("MIN_CURRENT_VOLUME")
                if min_cv is not None and float(min_cv) > 0.0:
                    k = ("min_cv", _kfloat(min_cv))
                    mask &= _get_mask(k, lambda: (arr["curr_vol"] >= float(min_cv)))

                max_cv = full_param_values.get("MAX_CURRENT_VOLUME")
                if max_cv is not None:
                    k = ("max_cv", _kfloat(max_cv))
                    mask &= _get_mask(k, lambda: (arr["curr_vol"] <= float(max_cv)))

                # volume_threshold_max (post)
                vtm = full_param_values.get("VOLUME_THRESHOLD_MAX")
                if vtm is not None:
                    k = ("vtm", _kfloat(vtm))
                    mask &= _get_mask(k, lambda: (arr["vol_ratio"] <= float(vtm)))

                # UTC hour range
                hmin = full_param_values.get("UTC_HOUR_MIN")
                hmax = full_param_values.get("UTC_HOUR_MAX")
                if hmin is not None and hmax is not None:
                    hk = (hmin, hmax)
                    parsed = hour_cache.get(hk)
                    if parsed is None:
                        try:
                            parsed = (int(hmin), int(hmax))
                        except Exception:
                            parsed = (None, None)
                        hour_cache[hk] = parsed
                    hhmin, hhmax = parsed
                    if hhmin is not None and hhmax is not None:
                        k = ("hour", hhmin, hhmax)
                        mask &= _get_mask(k, lambda: _utc_hour_mask(arr["utc_hour"], hhmin, hhmax))

                # weekday set (se usato)
                utc_weekday_raw = full_param_values.get("UTC_WEEKDAY")
                weekday_key = _stringify_utc_weekday(utc_weekday_raw)
                wset = weekday_cache.get(weekday_key)
                if wset is None and weekday_key:
                    wset = _parse_weekday_to_set(utc_weekday_raw)
                    weekday_cache[weekday_key] = wset
                if wset is not None and len(wset) > 0:
                    wk = tuple(sorted(list(wset)))
                    k = ("weekday", wk)

                    def _build_weekday_mask():
                        allowed = np.zeros(7, dtype=bool)
                        for wd in wk:
                            if 0 <= int(wd) <= 6:
                                allowed[int(wd)] = True
                        return allowed[arr["utc_weekday"]]

                    mask &= _get_mask(k, _build_weekday_mask)

                # price_ratio filter
                pr_mode = full_param_values.get("PRICE_RATIO_AT_SPIKE_MODE", "TAILS")
                pr_min = full_param_values.get("PRICE_RATIO_AT_SPIKE_MIN")
                pr_max = full_param_values.get("PRICE_RATIO_AT_SPIKE_MAX")
                k = ("pr", str(pr_mode).strip().upper(), _kfloat(pr_min), _kfloat(pr_max))
                mask &= _get_mask(k, lambda: _price_ratio_mask(arr["price_ratio"], pr_mode, pr_min, pr_max))

                # HL_MIN_PCT filter (post)
                hl_min = full_param_values.get("HL_MIN_PCT")
                if hl_min is not None:
                    k = ("hlmin", _kfloat(hl_min))
                    mask &= _get_mask(k, lambda: (arr["hl_pct"] >= float(hl_min)))

                # BODY_RATIO_MIN filter (post)
                br_min = full_param_values.get("BODY_RATIO_MIN")
                if br_min is not None:
                    k = ("brmin", _kfloat(br_min))
                    mask &= _get_mask(k, lambda: (arr["body_ratio"] >= float(br_min)))

                idx_sel = np.flatnonzero(mask)

            # ==== Scrivi file eventi (nome stabile) ====
            spikes_file = f"simulation_results/spikes_excel_opt_{combination_index}.csv"
            spikes_filename = os.path.basename(spikes_file)
            t_csv = time.perf_counter()
            spikes_batch.append((spikes_file, idx_sel))
            if len(spikes_batch) >= _SPIKES_BATCH_SIZE:
                _write_spikes_batch(arr, spikes_batch, dt_cache)
            stats["csv_write_s"] += (time.perf_counter() - t_csv)

            # ==== Metriche FAST (identiche ai campi che usi nel CSV finale) ====
            m = _compute_metrics_for_optimizer(arr, idx_sel)

            # riga per optimizer_results.csv
            row = {
                "VOLUME_LOOKBACK": full_param_values.get("VOLUME_LOOKBACK", 0),
                "VOLUME_THRESHOLD": full_param_values.get("VOLUME_THRESHOLD", 0),
                "VOLUME_THRESHOLD_MAX": full_param_values.get("VOLUME_THRESHOLD_MAX"),

                "PRICE_INCREASE_WINDOW": full_param_values.get("PRICE_INCREASE_WINDOW", 0),
                "MIN_PRICE_INCREASE_PERCENT": full_param_values.get("MIN_PRICE_INCREASE_PERCENT", 0),

                "MIN_CURRENT_VOLUME": full_param_values.get("MIN_CURRENT_VOLUME", 0),
                "MAX_CURRENT_VOLUME": full_param_values.get("MAX_CURRENT_VOLUME"),

                "SPIKE_COOLDOWN_CANDLES": full_param_values.get("SPIKE_COOLDOWN_CANDLES", 0),

                "UTC_HOUR_MIN": full_param_values.get("UTC_HOUR_MIN", ""),
                "UTC_HOUR_MAX": full_param_values.get("UTC_HOUR_MAX", ""),

                "UTC_WEEKDAY": _stringify_utc_weekday(full_param_values.get("UTC_WEEKDAY")),
                "PRICE_RATIO_AT_SPIKE_MODE": (
                    str(full_param_values.get("PRICE_RATIO_AT_SPIKE_MODE")).strip().upper()
                    if full_param_values.get("PRICE_RATIO_AT_SPIKE_MODE") is not None
                    else ""
                ),
                "PRICE_RATIO_AT_SPIKE_MIN": full_param_values.get("PRICE_RATIO_AT_SPIKE_MIN"),
                "PRICE_RATIO_AT_SPIKE_MAX": full_param_values.get("PRICE_RATIO_AT_SPIKE_MAX"),
                "HL_MIN_PCT": full_param_values.get("HL_MIN_PCT"),
                "BODY_RATIO_MIN": full_param_values.get("BODY_RATIO_MIN"),

                "SPIKES_FILE": spikes_filename,

                "efficiency_long": m["efficiency_long"],
                "avg_price_increase_long": m["avg_price_increase_long"],
                "avg_lead_time_long": m["avg_lead_time_long"],

                "efficiency_short": m["efficiency_short"],
                "avg_price_decrease_short": m["avg_price_decrease_short"],
                "avg_lead_time_short": m["avg_lead_time_short"],

                "total_spikes": m["total_spikes"],
            }

            results.append((combination_index, row))
            stats["soft_eval_s"] += (time.perf_counter() - t_soft)
            stats["rows"] += 1

        if spikes_batch:
            t_csv_flush = time.perf_counter()
            _write_spikes_batch(arr, spikes_batch, dt_cache)
            stats["csv_write_s"] += (time.perf_counter() - t_csv_flush)

    return {"rows": results, "stats": stats}


def main():
    range_config, output_csv, enabled_flags = load_optimizer_config("optimizer.env")

    active_params = [p for p in PARAMS if enabled_flags.get(p[0], False)]
    disabled_params = [p for p in PARAMS if not enabled_flags.get(p[0], False)]

    base_config = simulator.load_simulator_config()

    numba_requested = bool(base_config.get("core_use_numba", False))
    numba_available = bool(getattr(simulator, "NUMBA_AVAILABLE", False))
    numba_active = bool(numba_requested and numba_available)
    print(f"NUMBA ACTIVE={numba_active} (requested={numba_requested}, available={numba_available})")

    base_param_values = {}
    for env_name, config_key in ENV_TO_CONFIG_KEY.items():
        base_param_values[env_name] = base_config.get(config_key)

    hard_active_params = [(n, t) for (n, t) in active_params if n in HARD_PARAMS]
    soft_active_params = [(n, t) for (n, t) in active_params if n in SOFT_PARAMS]

    hard_value_lists = []
    for name, ptype in hard_active_params:
        if ptype is str:
            raise ValueError(f"Parametro HARD {name} attivo ma stringa: non supportato con MIN/MAX/STEP.")
        cfg = range_config[name]
        values = generate_values(cfg["min"], cfg["max"], cfg["step"], is_int=(ptype is int))
        hard_value_lists.append(values)

    soft_param_names = []
    soft_value_lists = []
    for name, ptype in soft_active_params:
        if ptype is str:
            raise ValueError(f"Parametro SOFT {name} attivo ma stringa: non supportato con MIN/MAX/STEP.")
        cfg = range_config[name]
        values = generate_values(cfg["min"], cfg["max"], cfg["step"], is_int=(ptype is int))
        soft_param_names.append(name)
        soft_value_lists.append(values)

    # combinazioni HARD
    hard_lengths = [len(v) for v in hard_value_lists]
    num_hard_groups = reduce(mul, hard_lengths, 1) if hard_lengths else 1

    # Precompute SOFT combos e rimuovi UTC_HOUR_MIN==UTC_HOUR_MAX (sanity)
    t_combo_build = time.perf_counter()
    if soft_value_lists:
        raw_soft_combos = list(itertools.product(*soft_value_lists))
    else:
        raw_soft_combos = [()]

    soft_combos = raw_soft_combos
    if "UTC_HOUR_MIN" in soft_param_names and "UTC_HOUR_MAX" in soft_param_names:
        idx_min = soft_param_names.index("UTC_HOUR_MIN")
        idx_max = soft_param_names.index("UTC_HOUR_MAX")
        filtered = []
        for combo in soft_combos:
            try:
                if int(combo[idx_min]) == int(combo[idx_max]):
                    continue
            except Exception:
                continue
            filtered.append(combo)
        soft_combos = filtered

    soft_total = len(soft_combos) if soft_combos else 1
    total_combinations = num_hard_groups * soft_total
    combo_build_s = (time.perf_counter() - t_combo_build)

    # performance
    workers_env = os.environ.get("OPTIMIZER_WORKERS", "").strip()
    if workers_env:
        try:
            num_workers = max(1, int(workers_env))
        except ValueError:
            num_workers = cpu_count() or 1
    else:
        num_workers = cpu_count() or 1

    chunk_env = os.environ.get("OPTIMIZER_CHUNK_SIZE", "auto").strip()
    chunk_size = _auto_chunk_size(num_hard_groups, num_workers, chunk_env)

    profile_enabled = os.environ.get("OPTIMIZER_PROFILE", "0").strip() == "1"
    spikes_batch_size = os.environ.get("OPTIMIZER_SPIKES_BATCH_SIZE", "1000").strip()

    print("========================================")
    print(" OPTIMIZER - GRID SEARCH PARAMETRI (CORE/SOFT) [FAST METRICS]")
    print("========================================")
    print("Parametri ATTIVI (HARD):")
    if hard_active_params:
        for (name, _), vals in zip(hard_active_params, hard_value_lists):
            print(f"  - {name}: {len(vals)} valori")
    else:
        print("  (nessun HARD attivo)")

    print("Parametri ATTIVI (SOFT):")
    if soft_active_params:
        for (name, _), vals in zip(soft_active_params, soft_value_lists):
            print(f"  - {name}: {len(vals)} valori")
    else:
        print("  (nessun SOFT attivo)")

    print("Parametri DISATTIVATI (da simulator.env):")
    if disabled_params:
        for name, _ in disabled_params:
            print(f"  - {name}")
    else:
        print("  (nessuno)")

    print("----------------------------------------")
    print(f"Gruppi HARD: {num_hard_groups}")
    print(f"Combinazioni SOFT (valide): {soft_total}")
    print(f"Totale combinazioni (valide): {total_combinations}")
    print(f"Output CSV: {output_csv}")
    print(f"OPTIMIZER_WORKERS: {num_workers}")
    print(f"OPTIMIZER_CHUNK_SIZE: {chunk_size} (source={chunk_env if chunk_env else 'auto'})")
    print(f"OPTIMIZER_PROFILE: {1 if profile_enabled else 0}")
    print(f"OPTIMIZER_SPIKES_BATCH_SIZE: {spikes_batch_size}")
    if profile_enabled:
        print(f"combo_build_s: {combo_build_s:.3f}")
    print("========================================\n")

    # HARD groups streaming
    if hard_value_lists:
        hard_combinations_iter = itertools.product(*hard_value_lists)
    else:
        hard_combinations_iter = [()]

    def hard_groups_iter():
        for hard_index, hard_values in enumerate(hard_combinations_iter):
            hard_full_params = dict(base_param_values)
            for (name, _ptype), value in zip(hard_active_params, hard_values):
                hard_full_params[name] = value
            yield (hard_index, hard_full_params)


    with open(output_csv, "w", encoding="utf-8", newline="", buffering=1024 * 1024) as out_f:
        out_writer = csv.writer(
            out_f,
            delimiter=";",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        _write_optimizer_header(out_writer)

        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(base_config, soft_param_names, soft_combos, soft_total, profile_enabled),
        ) as pool:
            hard_iter = iter(hard_groups_iter())
            dynamic_chunk_size = chunk_size
            max_in_flight = max(2, num_workers * 2)
            pending = []
            prof = {
                "chunks": 0,
                "hard_groups": 0,
                "rows": 0,
                "core_scan_s": 0.0,
                "array_build_s": 0.0,
                "soft_eval_s": 0.0,
                "csv_write_s": 0.0,
                "wall_chunks_s": 0.0,
                "core_load_np_s": 0.0,
                "core_extrema_s": 0.0,
                "core_candidate_build_s": 0.0,
                "core_candidate_loop_s": 0.0,
                "core_post_filter_s": 0.0,
                "core_price_prep_s": 0.0,
                "core_price_eval_s": 0.0,
                "core_datetime_s": 0.0,
            }

            def _submit_one():
                nonlocal dynamic_chunk_size
                chunk = _take_next_chunk(hard_iter, dynamic_chunk_size)
                if not chunk:
                    return False
                t0 = time.time()
                ar = pool.apply_async(_run_simulation_chunk, (chunk,))
                pending.append((ar, t0))
                return True

            for _ in range(max_in_flight):
                if not _submit_one():
                    break

            while pending:
                progressed = False
                i = 0
                while i < len(pending):
                    ar, t0 = pending[i]
                    if not ar.ready():
                        i += 1
                        continue

                    pending.pop(i)
                    chunk_out = ar.get()
                    elapsed_chunk = max(time.time() - t0, 1e-9)
                    dynamic_chunk_size = _adaptive_next_chunk_size(dynamic_chunk_size, elapsed_chunk)

                    chunk_rows = chunk_out.get("rows", []) if isinstance(chunk_out, dict) else chunk_out
                    chunk_stats = chunk_out.get("stats", {}) if isinstance(chunk_out, dict) else {}

                    for _, row in chunk_rows:
                        formatted = _format_optimizer_row_for_csv(row)
                        _write_optimizer_row(out_writer, formatted)

                    if profile_enabled:
                        prof["chunks"] += 1
                        prof["wall_chunks_s"] += elapsed_chunk
                        prof["hard_groups"] += int(chunk_stats.get("hard_groups", 0))
                        prof["rows"] += int(chunk_stats.get("rows", 0))
                        prof["core_scan_s"] += float(chunk_stats.get("core_scan_s", 0.0))
                        prof["array_build_s"] += float(chunk_stats.get("array_build_s", 0.0))
                        prof["soft_eval_s"] += float(chunk_stats.get("soft_eval_s", 0.0))
                        prof["csv_write_s"] += float(chunk_stats.get("csv_write_s", 0.0))
                        prof["core_load_np_s"] += float(chunk_stats.get("core_load_np_s", 0.0))
                        prof["core_extrema_s"] += float(chunk_stats.get("core_extrema_s", 0.0))
                        prof["core_candidate_build_s"] += float(chunk_stats.get("core_candidate_build_s", 0.0))
                        prof["core_candidate_loop_s"] += float(chunk_stats.get("core_candidate_loop_s", 0.0))
                        prof["core_post_filter_s"] += float(chunk_stats.get("core_post_filter_s", 0.0))
                        prof["core_price_prep_s"] += float(chunk_stats.get("core_price_prep_s", 0.0))
                        prof["core_price_eval_s"] += float(chunk_stats.get("core_price_eval_s", 0.0))
                        prof["core_datetime_s"] += float(chunk_stats.get("core_datetime_s", 0.0))

                    progressed = True
                    if len(pending) < max_in_flight:
                        _submit_one()

                if not progressed:
                    time.sleep(0.01)

        out_f.flush()

    if profile_enabled:
        total_accounted = combo_build_s + prof["core_scan_s"] + prof["array_build_s"] + prof["soft_eval_s"]
        print("\n----------- PROFILING (deterministico) -----------")
        print(f"chunks completati: {prof['chunks']}")
        print(f"hard_groups processati: {prof['hard_groups']}")
        print(f"righe/combinazioni emesse: {prof['rows']}")
        print(f"combo_build_s: {combo_build_s:.3f}")
        print(f"core_scan_s: {prof['core_scan_s']:.3f}")
        print(f"array_build_s: {prof['array_build_s']:.3f}")
        print(f"soft_eval_s (incl. csv+metriche): {prof['soft_eval_s']:.3f}")
        print(f"csv_write_s (subset di soft_eval): {prof['csv_write_s']:.3f}")
        print(f"wall_chunks_s (dispatcher observed): {prof['wall_chunks_s']:.3f}")
        print("-- core_scan sub-profile --")
        print(f"core_load_np_s: {prof['core_load_np_s']:.3f}")
        print(f"core_extrema_s: {prof['core_extrema_s']:.3f}")
        print(f"core_candidate_build_s: {prof['core_candidate_build_s']:.3f}")
        print(f"core_candidate_loop_s: {prof['core_candidate_loop_s']:.3f}")
        print(f"core_post_filter_s: {prof['core_post_filter_s']:.3f}")
        print(f"core_price_prep_s: {prof['core_price_prep_s']:.3f}")
        print(f"core_price_eval_s: {prof['core_price_eval_s']:.3f}")
        print(f"core_datetime_s: {prof['core_datetime_s']:.3f}")
        if prof['core_scan_s'] > 0:
            print(f"core_load_np_pct_of_core: {(prof['core_load_np_s'] / prof['core_scan_s']) * 100.0:.1f}%")
            print(f"core_extrema_pct_of_core: {(prof['core_extrema_s'] / prof['core_scan_s']) * 100.0:.1f}%")
            print(f"core_candidate_build_pct_of_core: {(prof['core_candidate_build_s'] / prof['core_scan_s']) * 100.0:.1f}%")
            print(f"core_candidate_loop_pct_of_core: {(prof['core_candidate_loop_s'] / prof['core_scan_s']) * 100.0:.1f}%")
            print(f"core_post_filter_pct_of_core: {(prof['core_post_filter_s'] / prof['core_scan_s']) * 100.0:.1f}%")
            print(f"core_price_prep_pct_of_core: {(prof['core_price_prep_s'] / prof['core_scan_s']) * 100.0:.1f}%")
            print(f"core_price_eval_pct_of_core: {(prof['core_price_eval_s'] / prof['core_scan_s']) * 100.0:.1f}%")
            print(f"core_datetime_pct_of_core: {(prof['core_datetime_s'] / prof['core_scan_s']) * 100.0:.1f}%")
        if total_accounted > 0:
            print(f"combo_build_pct: {(combo_build_s / total_accounted) * 100.0:.1f}%")
            print(f"core_scan_pct: {(prof['core_scan_s'] / total_accounted) * 100.0:.1f}%")
            print(f"array_build_pct: {(prof['array_build_s'] / total_accounted) * 100.0:.1f}%")
            print(f"soft_eval_pct: {(prof['soft_eval_s'] / total_accounted) * 100.0:.1f}%")
        print("--------------------------------------------------")

    print("\n========================================")
    print(" Ottimizzazione completata.")
    print(f" Risultati salvati in: {output_csv}")
    print("========================================")


if __name__ == "__main__":
    main()
