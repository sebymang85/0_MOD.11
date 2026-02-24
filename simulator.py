import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import time
import psutil
import traceback
import csv
from collections import deque
import importlib.util

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
if NUMBA_AVAILABLE:
    from numba import njit

warnings.filterwarnings('ignore')

# Configurazione logging DETTAGLIATA
def setup_detailed_logging(log_level='INFO', log_to_file=True, log_file_path='simulator_detailed.log'):
    """Configura logging iperdettagliato"""
    log_level = getattr(logging, log_level.upper())

    # Formatter dettagliato
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(detailed_formatter)

    handlers = [console_handler]

    # Handler file (opzionale)
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File sempre DEBUG
        file_handler.setFormatter(detailed_formatter)
        handlers.append(file_handler)

    # Configura logger root
    logging.basicConfig(
        level=logging.DEBUG,  # Livello più basso per catturare tutto
        handlers=handlers,
        force=True  # garantisce la riconfigurazione anche se il logger è già stato inizializzato
    )

    # Disabilita log troppo verbosi di alcune librerie
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    # Restituisce il logger configurato
    return logging.getLogger(__name__)

# INIZIALIZZA SUBITO IL LOGGER (NO FILE DI LOG ALL'IMPORT)
logger = setup_detailed_logging('INFO', log_to_file=False)


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _build_candidates_numba(vol, cvol, close, cclose, open_, high, low, lb, max_index, volume_threshold, cooldown_step):
        n_max = max_index - lb
        if n_max <= 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        cand_idx = np.empty(n_max, dtype=np.int64)
        cand_vr = np.empty(n_max, dtype=np.float64)
        cand_cv = np.empty(n_max, dtype=np.float64)
        cand_ap = np.empty(n_max, dtype=np.float64)
        cand_pr = np.empty(n_max, dtype=np.float64)
        hl_pct = np.empty(n_max, dtype=np.float64)
        body_ratio = np.empty(n_max, dtype=np.float64)

        out_n = 0
        next_allowed = lb

        for i in range(lb, max_index):
            if cooldown_step > 0 and i < next_allowed:
                continue

            curr_vol = vol[i]
            avg_vol = (cvol[i] - cvol[i - lb]) / lb
            if avg_vol <= 0.0 or curr_vol <= 0.0:
                continue

            vr = (curr_vol / avg_vol) * 100.0
            if vr < volume_threshold or vr > 100000000.0:
                continue

            cp = close[i]
            ap = (cclose[i] - cclose[i - lb]) / lb
            if cp <= 0.0 or ap <= 0.0:
                continue

            pr = ((cp - ap) / ap) * 100.0
            hi = high[i]
            lo = low[i]
            op = open_[i]
            hl = hi - lo

            hlv = 0.0
            if cp > 0.0:
                hlv = (hl / cp) * 100.0

            br = 0.0
            if hl > 0.0:
                diff = cp - op
                if diff < 0.0:
                    diff = -diff
                br = diff / hl

            cand_idx[out_n] = i
            cand_vr[out_n] = vr
            cand_cv[out_n] = curr_vol
            cand_ap[out_n] = ap
            cand_pr[out_n] = pr
            hl_pct[out_n] = hlv
            body_ratio[out_n] = br
            out_n += 1

            if cooldown_step > 0:
                next_allowed = i + cooldown_step

        return (
            cand_idx[:out_n],
            cand_vr[:out_n],
            cand_cv[:out_n],
            cand_ap[:out_n],
            cand_pr[:out_n],
            hl_pct[:out_n],
            body_ratio[:out_n],
        )


def _compute_candidate_future_extrema_numpy(
    high: np.ndarray,
    low: np.ndarray,
    cand_idx: np.ndarray,
    eval_window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute candidate future max/min and lead times on [idx+1, idx+eval_window]."""
    n = cand_idx.size
    max_future_price = np.zeros(n, dtype=np.float64)
    min_future_price = np.zeros(n, dtype=np.float64)
    lead_time_up = np.zeros(n, dtype=np.int32)
    lead_time_down = np.zeros(n, dtype=np.int32)

    if n == 0 or eval_window <= 0:
        return max_future_price, lead_time_up, min_future_price, lead_time_down

    high_n = high.shape[0]
    for j in range(n):
        i = int(cand_idx[j])
        start = i + 1
        end = i + eval_window + 1
        if start >= high_n:
            continue
        if end > high_n:
            end = high_n
        if start >= end:
            continue

        best_max = high[start]
        best_min = low[start]
        best_max_k = 0
        best_min_k = 0

        for k in range(1, end - start):
            hi = high[start + k]
            lo = low[start + k]

            # Tie behavior must match np.argmax/np.argmin: first occurrence wins.
            if hi > best_max:
                best_max = hi
                best_max_k = k
            if lo < best_min:
                best_min = lo
                best_min_k = k

        max_future_price[j] = best_max
        min_future_price[j] = best_min
        lead_time_up[j] = np.int32(best_max_k + 1)
        lead_time_down[j] = np.int32(best_min_k + 1)

    return max_future_price, lead_time_up, min_future_price, lead_time_down


if NUMBA_AVAILABLE:
    _compute_candidate_future_extrema_numba = njit(cache=True)(_compute_candidate_future_extrema_numpy)

class TradingSimulator:
    def __init__(self, config):
        self.config = config

        # D6=B FIX: base dir dati storico configurabile (per esecuzione in cwd trial)
        self.data_base_dir = config.get('data_base_dir', 'data/historical')

        # MEMMAP: base dir dati memmap configurabile (fallback a data/memmap)
        self.data_memmap_dir = config.get('data_memmap_dir', 'data/memmap')
        # MEMMAP: usa memmap se presente (fallback a parquet se non disponibile)
        self.use_memmap = config.get('use_memmap', True)
        self.symbol_np_cache = {}

        self.volume_lookback = config['volume_lookback']
        self.volume_threshold = config['volume_threshold']
        self.volume_threshold_max = config.get('volume_threshold_max')
        self.price_increase_window = config['price_increase_window']
        self.min_price_increase_percent = config['min_price_increase_percent']
        self.timeframe = config.get('timeframe', '1h')

        # NUOVO PARAMETRO: Min Current Volume
        self.min_current_volume = config.get('min_current_volume', 0)

        # NUOVO PARAMETRO: Max Current Volume (se None -> nessun limite)
        self.max_current_volume = config.get('max_current_volume')

        # Policy: ERROR + STOP RUN se MIN > MAX (entrambi settati / MAX non None)
        if self.max_current_volume is not None and self.min_current_volume > self.max_current_volume:
            raise ValueError(
                f"Configurazione non valida: MIN_CURRENT_VOLUME ({self.min_current_volume}) "
                f"> MAX_CURRENT_VOLUME ({self.max_current_volume}). Correggi simulator.env e riprova."
            )

        # NUOVO FILTRO: Range ORARIO UTC (post-qualifica)
        # UTC_HOUR_MIN: incluso
        # UTC_HOUR_MAX: escluso
        self.utc_hour_min = config.get('utc_hour_min')
        self.utc_hour_max = config.get('utc_hour_max')

        # Policy: ERROR + STOP RUN se solo uno dei due è settato
        if (self.utc_hour_min is None) != (self.utc_hour_max is None):
            raise ValueError(
                "Configurazione non valida: devi settare sia UTC_HOUR_MIN sia UTC_HOUR_MAX (o nessuno dei due). "
                "Correggi simulator.env e riprova."
            )

        # Policy: validazione range e ambiguità MIN==MAX (range [MIN,MAX) nullo)
        if self.utc_hour_min is not None and self.utc_hour_max is not None:
            if not (0 <= int(self.utc_hour_min) <= 23) or not (0 <= int(self.utc_hour_max) <= 23):
                raise ValueError(
                    f"Configurazione non valida: UTC_HOUR_MIN ({self.utc_hour_min}) e UTC_HOUR_MAX ({self.utc_hour_max}) "
                    f"devono essere tra 0 e 23. Correggi simulator.env e riprova."
                )
            if int(self.utc_hour_min) == int(self.utc_hour_max):
                raise ValueError(
                    f"Configurazione ambigua: UTC_HOUR_MIN ({self.utc_hour_min}) == UTC_HOUR_MAX ({self.utc_hour_max}). "
                    "Con range [MIN,MAX) il risultato è nullo/ambiguo. Correggi simulator.env e riprova."
                )

        # NUOVO FILTRO: Weekday UTC (lista esplicita) - post-qualifica
        # 0 = Lunedì ... 6 = Domenica
        utc_weekday_list = config.get('utc_weekday')
        if utc_weekday_list is None:
            self.utc_weekday_set = None
        else:
            # Normalizza + valida (difensivo, anche se già validato in load_simulator_config)
            try:
                normalized = [int(x) for x in utc_weekday_list]
            except Exception:
                raise ValueError(
                    f"Configurazione non valida: UTC_WEEKDAY deve essere lista di interi [0..6]. Valore: {utc_weekday_list}"
                )
            if not normalized:
                raise ValueError(
                    "Configurazione non valida: UTC_WEEKDAY è vuoto dopo parsing. Correggi simulator.env e riprova."
                )
            for d in normalized:
                if d < 0 or d > 6:
                    raise ValueError(
                        f"Configurazione non valida: UTC_WEEKDAY contiene valore fuori range [0..6]: {d}. "
                        "Correggi simulator.env e riprova."
                    )
            self.utc_weekday_set = set(normalized)

        # PARAMETRO MODALITÀ FILTRO price_ratio_at_spike
        # - TAILS (default): logica attuale (code)
        # - RANGE: range tra MIN e MAX
        self.price_ratio_at_spike_mode = config.get('price_ratio_at_spike_mode', 'TAILS')
        if self.price_ratio_at_spike_mode is None:
            self.price_ratio_at_spike_mode = 'TAILS'
        self.price_ratio_at_spike_mode = str(self.price_ratio_at_spike_mode).strip().upper()

        if self.price_ratio_at_spike_mode not in ('TAILS', 'RANGE'):
            raise ValueError(
                f"Configurazione non valida: PRICE_RATIO_AT_SPIKE_MODE deve essere TAILS o RANGE. "
                f"Valore: {self.price_ratio_at_spike_mode}"
            )

        # NUOVI PARAMETRI: filtri su price_ratio_at_spike (%)
        # TAILS:
        #   - MIN: tiene spike con price_ratio_at_spike <= MIN  (coda sinistra: -∞ → MIN)
        #   - MAX: tiene spike con price_ratio_at_spike >= MAX  (coda destra: MAX → +∞)
        #   - se entrambi settati → condizione OR.
        # RANGE:
        #   - tiene spike con MIN <= price_ratio_at_spike <= MAX (inclusivo)
        self.price_ratio_at_spike_min = config.get('price_ratio_at_spike_min')
        self.price_ratio_at_spike_max = config.get('price_ratio_at_spike_max')

        # Validazioni difensive per MODE=RANGE (mantenendo input attuali MIN/MAX)
        if self.price_ratio_at_spike_mode == 'RANGE':
            if self.price_ratio_at_spike_min is None or self.price_ratio_at_spike_max is None:
                raise ValueError(
                    "Configurazione non valida: con PRICE_RATIO_AT_SPIKE_MODE=RANGE devi settare "
                    "sia PRICE_RATIO_AT_SPIKE_MIN sia PRICE_RATIO_AT_SPIKE_MAX. Correggi simulator.env e riprova."
                )
            if float(self.price_ratio_at_spike_min) >= float(self.price_ratio_at_spike_max):
                raise ValueError(
                    f"Configurazione non valida: con PRICE_RATIO_AT_SPIKE_MODE=RANGE devi avere "
                    f"PRICE_RATIO_AT_SPIKE_MIN ({self.price_ratio_at_spike_min}) < PRICE_RATIO_AT_SPIKE_MAX ({self.price_ratio_at_spike_max}). "
                    "Correggi simulator.env e riprova."
                )

        # NUOVI PARAMETRI: filtri su candela spike (ampiezza e body)
        # HL_MIN_PCT: (high-low)/close*100 minimo richiesto sulla candela spike
        # BODY_RATIO_MIN: abs(close-open)/(high-low) minimo richiesto sulla candela spike
        self.hl_min_pct = config.get('hl_min_pct')
        self.body_ratio_min = config.get('body_ratio_min')

        if self.hl_min_pct is not None and float(self.hl_min_pct) < 0:
            raise ValueError(
                f"Configurazione non valida: HL_MIN_PCT ({self.hl_min_pct}) deve essere >= 0. Correggi simulator.env e riprova."
            )
        if self.body_ratio_min is not None:
            br = float(self.body_ratio_min)
            if br < 0 or br > 1:
                raise ValueError(
                    f"Configurazione non valida: BODY_RATIO_MIN ({self.body_ratio_min}) deve essere tra 0 e 1. Correggi simulator.env e riprova."
                )

        # Performance optimization - CONFIGURABILE
        self.max_workers = config.get('max_workers', min(mp.cpu_count(), 12))
        self.batch_size = config.get('batch_size', 25)
        self.symbol_data_cache = {}
        self._dt_utc_cache = {}
        # Cache leggera per finestre forward CORE (A2):
        # key=(symbol, eval_window, len(high), id(high), id(low))
        # value={highs_fw, lows_fw}
        self._core_fw_cache = {}

        # NUOVO: forza esecuzione sequenziale (usato quando chiamato da optimizer multiprocess)
        self.force_sequential = config.get('force_sequential', False)
        self.core_extrema_mode = str(config.get('core_extrema_mode', 'candidate')).strip().lower()
        if self.core_extrema_mode not in ('candidate', 'precompute'):
            self.core_extrema_mode = 'candidate'
        self.core_use_numba = bool(config.get('core_use_numba', NUMBA_AVAILABLE))

        # Statistics tracking
        self.symbols_processed = 0
        self.total_symbols = 0
        self.start_time = None

        # NUOVO: cooldown spike (in candele)
        self.spike_cooldown_candles = config.get('spike_cooldown_candles', 0)

        logger.info(f"🔧 SIMULATOR CONFIGURATO:")
        logger.info(f"   • Data Base Dir: {self.data_base_dir}")
        logger.info(f"   • Data Memmap Dir: {self.data_memmap_dir}")
        logger.info(f"   • Use memmap: {self.use_memmap}")
        logger.info(f"   • Workers: {self.max_workers}")
        logger.info(f"   • Batch size: {self.batch_size}")
        logger.info(f"   • Volume Lookback: {self.volume_lookback}")
        logger.info(f"   • Volume Threshold: {self.volume_threshold}")
        logger.info(f"   • Volume Threshold MAX: {self.volume_threshold_max}")
        logger.info(f"   • Price Window: {self.price_increase_window}")
        logger.info(f"   • Min Price Increase: {self.min_price_increase_percent}%")
        logger.info(f"   • Min Current Volume: {self.min_current_volume}")
        logger.info(f"   • Max Current Volume: {self.max_current_volume}")
        logger.info(f"   • UTC Hour MIN (incl): {self.utc_hour_min}")
        logger.info(f"   • UTC Hour MAX (escl): {self.utc_hour_max}")
        logger.info(f"   • UTC Weekday (allowed): {sorted(list(self.utc_weekday_set)) if self.utc_weekday_set is not None else None}")
        logger.info(f"   • Price Ratio @ Spike MODE: {self.price_ratio_at_spike_mode}")
        logger.info(f"   • Price Ratio @ Spike MIN: {self.price_ratio_at_spike_min}")
        logger.info(f"   • Price Ratio @ Spike MAX: {self.price_ratio_at_spike_max}")
        logger.info(f"   • HL_MIN_PCT: {self.hl_min_pct}")
        logger.info(f"   • BODY_RATIO_MIN: {self.body_ratio_min}")
        logger.info(f"   • Spike Cooldown (candles): {self.spike_cooldown_candles}")
        logger.info(f"   • Force sequential: {self.force_sequential}")
        logger.info(f"   • Core extrema mode: {self.core_extrema_mode}")
        logger.info(f"   • Core numba enabled: {self.core_use_numba and NUMBA_AVAILABLE}")

    # ============================
    # MEMMAP LOADER
    # ============================
    def _get_memmap_symbol_dir(self, symbol: str) -> str:
        """
        Supporta due layout:
          1) <DATA_MEMMAP_DIR>/<timeframe>/<symbol>   (come il tuo converter)
          2) <DATA_MEMMAP_DIR>/<symbol>/<timeframe>   (layout "parquet-like")
        Ritorna il primo che esiste, altrimenti stringa vuota.
        """
        if not self.data_memmap_dir:
            return ""

        d1 = os.path.join(self.data_memmap_dir, str(self.timeframe), str(symbol))
        if os.path.isdir(d1):
            return d1

        d2 = os.path.join(self.data_memmap_dir, str(symbol), str(self.timeframe))
        if os.path.isdir(d2):
            return d2

        return ""

    def _memmap_files_exist(self, symbol: str) -> bool:
        d = self._get_memmap_symbol_dir(symbol)
        if not d:
            return False

        expected = [
            "timestamp.npy",
            "open.npy",
            "high.npy",
            "low.npy",
            "close.npy",
            "volume.npy",
            "cvol.npy",
            "cclose.npy",
        ]
        for f in expected:
            if not os.path.exists(os.path.join(d, f)):
                return False
        return True

    def load_symbol_numpy_data(self, symbol: str) -> Dict[str, Any]:
        """
        Ritorna np_data nel formato usato dal simulator:
          ts, vol, close, high, low, open, cvol, cclose
        Prova MEMMAP (np.load mmap_mode='r'), fallback a parquet->DataFrame->_prepare_symbol_numpy.
        Cache per simbolo (memmap o numpy arrays).
        """
        cached = self.symbol_np_cache.get(symbol)
        if cached is not None:
            return cached

        # Fast-path RAM: se il DataFrame è già in cache, evita I/O MEMMAP/parquet.
        # Questo è particolarmente utile nell'optimizer, dove i simboli vengono
        # pre-caricati nel worker init.
        df_cached = self.symbol_data_cache.get(symbol)
        if df_cached is not None:
            np_data = self._prepare_symbol_numpy(df_cached)
            self.symbol_np_cache[symbol] = np_data
            return np_data

        # MEMMAP path (se abilitato e presente)
        if self.use_memmap and self._memmap_files_exist(symbol):
            try:
                d = self._get_memmap_symbol_dir(symbol)

                ts = np.load(os.path.join(d, "timestamp.npy"), mmap_mode="r")
                op = np.load(os.path.join(d, "open.npy"), mmap_mode="r")
                hi = np.load(os.path.join(d, "high.npy"), mmap_mode="r")
                lo = np.load(os.path.join(d, "low.npy"), mmap_mode="r")
                cl = np.load(os.path.join(d, "close.npy"), mmap_mode="r")
                vol = np.load(os.path.join(d, "volume.npy"), mmap_mode="r")
                cvol = np.load(os.path.join(d, "cvol.npy"), mmap_mode="r")
                cclose = np.load(os.path.join(d, "cclose.npy"), mmap_mode="r")

                np_data = {
                    'ts': ts,
                    'vol': vol,
                    'close': cl,
                    'high': hi,
                    'low': lo,
                    'open': op,
                    'cvol': cvol,
                    'cclose': cclose
                }

                self.symbol_np_cache[symbol] = np_data
                return np_data

            except Exception as e:
                logger.error(f"❌ Errore caricamento MEMMAP {symbol}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                # fallback a parquet

        # Fallback parquet
        df = self.load_symbol_data(symbol)
        np_data = self._prepare_symbol_numpy(df)
        self.symbol_np_cache[symbol] = np_data
        return np_data

    def _get_utc_hour_from_timestamp_ms(self, timestamp_ms: Any) -> int:
        """Estrae l'ora UTC (0-23) da timestamp epoch in millisecondi."""
        try:
            ts_int = int(timestamp_ms)
        except Exception:
            raise ValueError(f"Timestamp non valido (atteso epoch ms): {timestamp_ms}")

        return int((ts_int // 3600000) % 24)

    def _get_utc_weekday_from_timestamp_ms(self, timestamp_ms: Any) -> int:
        """Estrae il weekday UTC (0=Lunedì .. 6=Domenica) da timestamp epoch in millisecondi."""
        try:
            ts_int = int(timestamp_ms)
        except Exception:
            raise ValueError(f"Timestamp non valido (atteso epoch ms): {timestamp_ms}")

        # 1970-01-01 = Giovedì -> weekday=3 (Monday=0)
        return int(((ts_int // 86400000) + 3) % 7)

    def _is_in_utc_hour_range(self, timestamp_ms: Any) -> bool:
        """
        Verifica filtro orario UTC con semantica [MIN, MAX):
          - MIN incluso
          - MAX escluso
        Supporta wrap mezzanotte quando MIN > MAX.
        Se MIN/MAX non settati => nessun filtro (True).
        """
        if self.utc_hour_min is None or self.utc_hour_max is None:
            return True

        hour = self._get_utc_hour_from_timestamp_ms(timestamp_ms)
        hmin = int(self.utc_hour_min)
        hmax = int(self.utc_hour_max)

        # Caso normale
        if hmin < hmax:
            return (hour >= hmin) and (hour < hmax)

        # Caso wrap mezzanotte (es. 22 -> 2)
        return (hour >= hmin) or (hour < hmax)

    def _is_in_utc_weekday_set(self, timestamp_ms: Any) -> bool:
        """
        Verifica filtro weekday UTC con semantica "membership":
          - Se UTC_WEEKDAY non settato => nessun filtro (True)
          - Altrimenti tiene solo weekday presenti nella lista (0=Lun .. 6=Dom)
        """
        if self.utc_weekday_set is None:
            return True

        wd = self._get_utc_weekday_from_timestamp_ms(timestamp_ms)
        return wd in self.utc_weekday_set

    def get_available_symbols(self) -> List[str]:
        """Restituisce lista simboli disponibili nel database storico"""
        historical_dir = self.data_base_dir
        symbols = []

        if not os.path.exists(historical_dir):
            raise FileNotFoundError(f"Directory {historical_dir} non trovata")

        for item in os.listdir(historical_dir):
            item_path = os.path.join(historical_dir, item)
            if os.path.isdir(item_path):
                parquet_path = os.path.join(item_path, f"{self.timeframe}.parquet")
                if os.path.exists(parquet_path):
                    symbols.append(item)

        self.total_symbols = len(symbols)
        logger.info(f"📁 Trovati {self.total_symbols} simboli con dati storici")
        return symbols

    def check_timeframe_data_exists(self) -> bool:
        """Verifica se i dati per il timeframe configurato esistono"""
        symbols = self.get_available_symbols()

        if not symbols:
            logger.error("❌ Nessun simbolo trovato nel database")
            return False

        # Controlla il primo simbolo come campione
        sample_symbol = symbols[0]
        expected_path = f"{self.data_base_dir}/{sample_symbol}/{self.timeframe}.parquet"

        if not os.path.exists(expected_path):
            # fallback memmap se presente
            if self.use_memmap and self._memmap_files_exist(sample_symbol):
                logger.info(f"✅ Timeframe {self.timeframe} verificato - memmap presenti (parquet mancante)")
                return True

            logger.error(f"❌ TIMEFRAME NON TROVATO: {self.timeframe}")
            logger.error(f"   Percorso atteso: {expected_path}")

            # Cerca timeframe disponibili
            symbol_dir = f"{self.data_base_dir}/{sample_symbol}"
            available_files = []
            if os.path.exists(symbol_dir):
                for file in os.listdir(symbol_dir):
                    if file.endswith('.parquet'):
                        available_files.append(file.replace('.parquet', ''))

            if available_files:
                logger.error(f"   Timeframe disponibili: {available_files}")
                logger.error(f"   Modifica TIMEFRAME in simulator.env")
            else:
                logger.error(f"   Nessun dato trovato per {sample_symbol}")
                logger.error(f"   Esegui prima: python backtest_orchestrator.py")

            return False

        logger.info(f"✅ Timeframe {self.timeframe} verificato - dati presenti")
        return True

    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Carica dati storici per un simbolo con cache e logica corretta"""
        logger.debug(f"Caricamento dati per {symbol}")

        if symbol in self.symbol_data_cache:
            logger.debug(f"Cache hit per {symbol}")
            return self.symbol_data_cache[symbol]

        file_path = f"{self.data_base_dir}/{symbol}/{self.config['timeframe']}.parquet"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dati non trovati per {symbol}")

        try:
            load_start = time.time()
            df = pd.read_parquet(file_path)
            load_time = time.time() - load_start

            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Colonne mancanti: {missing_columns}")

            # 🔥 ORDINAMENTO: ascending=True (timestamp crescente)
            df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

            self.symbol_data_cache[symbol] = df

            logger.debug(f"Dati caricati per {symbol}: {len(df)} candele in {load_time:.2f}s")
            return df

        except Exception as e:
            logger.error(f"Errore caricamento dati {symbol}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    # ============================
    # FIX B: PREPARAZIONE NUMPY
    # ============================
    def _prepare_symbol_numpy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepara array numpy + cumulative sums per evitare slicing pandas nel loop.
        Mantiene la stessa logica (medie su lookback escluso current candle).
        """
        ts = df['timestamp'].to_numpy()
        vol = df['volume'].to_numpy(dtype=float)
        close = df['close'].to_numpy(dtype=float)
        high = df['high'].to_numpy(dtype=float)
        low = df['low'].to_numpy(dtype=float)
        open_ = df['open'].to_numpy(dtype=float)

        # Prefix sums con elemento 0 iniziale:
        # c[k] = somma degli elementi [0 .. k-1]
        cvol = np.concatenate(([0.0], np.cumsum(vol)))
        cclose = np.concatenate(([0.0], np.cumsum(close)))

        return {
            'ts': ts,
            'vol': vol,
            'close': close,
            'high': high,
            'low': low,
            'open': open_,
            'cvol': cvol,
            'cclose': cclose
        }

    def detect_volume_spike_core_vectorized(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """
        (Legacy) Rileva SPIKE CORE (a monte) per gestione cooldown:
        - Ignora filtri MIN_CURRENT_VOLUME / MAX_CURRENT_VOLUME
        - Ignora filtro VOLUME_THRESHOLD_MAX (cooldown deve scattare comunque anche se ratio troppo alto)
        - Applica solo: volume_ratio >= VOLUME_THRESHOLD + sanity checks
        """
        if current_index < self.volume_lookback:
            return None

        try:
            # Estrai le candele per il lookback
            lookback_start = current_index - self.volume_lookback
            lookback_end = current_index
            lookback_candles = df.iloc[lookback_start:lookback_end]
            current_candle = df.iloc[current_index]

            # Calcola volume corrente e medio - VETTORIZZATO
            current_volume = float(current_candle['volume'])
            lookback_volumes = lookback_candles['volume'].astype(float).values

            # 0 = nessuna attività -> non può essere spike
            if current_volume <= 0:
                logger.debug(f"📉 [CORE] Volume corrente = {current_volume:.0f} (<=0) - evento nullo, skip")
                return None

            # calcolo media lookback includendo gli zeri (zeri = reali)
            average_volume = float(np.mean(lookback_volumes))

            # se baseline è 0 (lookback tutto 0) non posso definire un ratio utile
            if average_volume <= 0:
                logger.debug("📉 [CORE] Average volume = 0 nel lookback (nessuna attività) - skip")
                return None

            volume_ratio = current_volume / average_volume

            # VERIFICA DATI RAGIONEVOLI - FIX PER VALORI IMPOSSIBILI
            if volume_ratio > 100000000:  # Threshold molto alto
                logger.warning(f"⚠️  [CORE] Volume ratio IMPROBABILE: {volume_ratio:.2f} per candela {current_index}")
                logger.warning(f"   Current volume: {current_volume}, Avg volume: {average_volume}")
                return None  # Salta spike improbabile

            # SPIKE CORE: solo threshold minimo
            # OTTIMIZZAZIONE: calcola i prezzi SOLO se supera la soglia volume
            if volume_ratio >= self.volume_threshold:
                # Calcola prezzo medio lookback e prezzo corrente - VETTORIZZATO
                lookback_prices = lookback_candles['close'].astype(float).values
                current_price = float(current_candle['close'])
                average_price = np.mean(lookback_prices)

                # VERIFICA PREZZI RAGIONEVOLI
                if current_price <= 0 or average_price <= 0:
                    logger.warning(f"⚠️  [CORE] Prezzo negativo o zero: current={current_price}, avg={average_price}")
                    return None

                price_ratio = ((current_price - average_price) / average_price) * 100

                logger.debug(f"🔍 [CORE] Candela {current_index}: volume_ratio={volume_ratio:.2f}, price_ratio={price_ratio:.2f}%")

                spike_info = {
                    'symbol': 'TEMP',  # Sarà sovrascritto
                    'timestamp': current_candle['timestamp'],
                    'datetime': current_candle.get('datetime', pd.Timestamp(current_candle['timestamp'], unit='ms')),
                    'volume_ratio': volume_ratio,
                    'current_volume': current_volume,
                    'average_volume': average_volume,
                    'current_price': current_price,
                    'average_price': average_price,
                    'price_ratio': price_ratio,
                    'candle_index': current_index,
                    'lookback_period': self.volume_lookback
                }
                logger.debug(f"🚨 [CORE] SPIKE RILEVATO: ratio={volume_ratio:.2f}, price={current_price}, avg_price={average_price:.4f}")
                return spike_info

            return None

        except Exception as e:
            logger.error(f"❌ Errore rilevamento spike CORE vettorizzato: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def detect_volume_spike_core_numpy(self, np_data: Dict[str, Any], current_index: int) -> Dict[str, Any]:
        """
        FIX B: versione array-based del CORE spike detection.
        Logica IDENTICA alla versione pandas:
        - lookback = [current_index - lookback, current_index) (current candle esclusa)
        - ignora min/max volume, ignora ratio_max
        - richiede: volume_ratio >= volume_threshold + sanity checks
        """
        if current_index < self.volume_lookback:
            return None

        try:
            ts = np_data['ts']
            vol = np_data['vol']
            close = np_data['close']
            cvol = np_data['cvol']
            cclose = np_data['cclose']

            i = current_index
            lb = self.volume_lookback

            current_volume = float(vol[i])

            # 0 = nessuna attività -> non può essere spike
            if current_volume <= 0:
                logger.debug(f"📉 [CORE] Volume corrente = {current_volume:.0f} (<=0) - evento nullo, skip")
                return None

            # media lookback includendo zeri (zeri = reali)
            # somma lookback: [i-lb, i)
            sum_vol = float(cvol[i] - cvol[i - lb])
            average_volume = sum_vol / float(lb)

            # se baseline è 0 (lookback tutto 0) non posso definire un ratio utile
            if average_volume <= 0:
                logger.debug("📉 [CORE] Average volume = 0 nel lookback (nessuna attività) - skip")
                return None

            volume_ratio = current_volume / average_volume

            # VERIFICA DATI RAGIONEVOLI - FIX PER VALORI IMPOSSIBILI
            if volume_ratio > 100000000:
                logger.warning(f"⚠️  [CORE] Volume ratio IMPROBABILE: {volume_ratio:.2f} per candela {current_index}")
                logger.warning(f"   Current volume: {current_volume}, Avg volume: {average_volume}")
                return None

            # SPIKE CORE: solo threshold minimo
            # calcola i prezzi SOLO se supera la soglia volume
            if volume_ratio >= self.volume_threshold:
                current_price = float(close[i])

                # media prezzi lookback: [i-lb, i)
                sum_close = float(cclose[i] - cclose[i - lb])
                average_price = sum_close / float(lb)

                # VERIFICA PREZZI RAGIONEVOLI
                if current_price <= 0 or average_price <= 0:
                    logger.warning(f"⚠️  [CORE] Prezzo negativo o zero: current={current_price}, avg={average_price}")
                    return None

                price_ratio = ((current_price - average_price) / average_price) * 100

                logger.debug(f"🔍 [CORE] Candela {current_index}: volume_ratio={volume_ratio:.2f}, price_ratio={price_ratio:.2f}%")

                ts_i = ts[i]
                spike_info = {
                    'symbol': 'TEMP',  # Sarà sovrascritto
                    'timestamp': ts_i,
                    'datetime': pd.Timestamp(ts_i, unit='ms'),
                    'volume_ratio': volume_ratio,
                    'current_volume': current_volume,
                    'average_volume': average_price if False else average_volume,  # mantiene chiave identica, no side effects
                    'current_price': current_price,
                    'average_price': average_price,
                    'price_ratio': price_ratio,
                    'candle_index': current_index,
                    'lookback_period': self.volume_lookback
                }
                logger.debug(f"🚨 [CORE] SPIKE RILEVATO: ratio={volume_ratio:.2f}, price={current_price}, avg_price={average_price:.4f}")
                return spike_info

            return None

        except Exception as e:
            logger.error(f"❌ Errore rilevamento spike CORE numpy: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def _qualify_spike_post_filters(self, spike_info: Dict[str, Any]) -> bool:
        """
        Applica filtri "a valle" sullo spike core:
        - MIN_CURRENT_VOLUME / MAX_CURRENT_VOLUME
        - VOLUME_THRESHOLD_MAX (solo qualifica, NON influenza cooldown)
        - UTC_HOUR_MIN / UTC_HOUR_MAX (range [MIN,MAX), post-qualifica, NON influenza cooldown)
        - UTC_WEEKDAY (lista esplicita, post-qualifica, NON influenza cooldown)
        - HL_MIN_PCT (ampiezza candela spike: (high-low)/close*100)
        - BODY_RATIO_MIN (body/(high-low): abs(close-open)/(high-low))
        """
        if not spike_info:
            return False

        current_volume = float(spike_info.get('current_volume', 0.0))
        volume_ratio = float(spike_info.get('volume_ratio', 0.0))

        # FILTRO: Volume corrente minimo
        if current_volume < self.min_current_volume:
            logger.debug(f"📉 [QUALIFY] Volume corrente {current_volume:.0f} sotto soglia {self.min_current_volume:.0f} - scarto")
            return False

        # FILTRO: Volume corrente massimo (inclusivo)
        if self.max_current_volume is not None and current_volume > self.max_current_volume:
            logger.debug(f"📈 [QUALIFY] Volume corrente {current_volume:.0f} sopra soglia {self.max_current_volume:.0f} - scarto")
            return False

        # FILTRO: Volume ratio massimo (se settato) - solo qualifica
        if self.volume_threshold_max is not None and volume_ratio > self.volume_threshold_max:
            logger.debug(f"📈 [QUALIFY] Volume ratio {volume_ratio:.2f} sopra soglia MAX {self.volume_threshold_max:.2f} - scarto")
            return False

        # FILTRO: Fascia oraria UTC (range [MIN,MAX))
        if self.utc_hour_min is not None and self.utc_hour_max is not None:
            ts = spike_info.get('timestamp')
            if not self._is_in_utc_hour_range(ts):
                try:
                    hour = self._get_utc_hour_from_timestamp_ms(ts)
                    logger.debug(
                        f"🕒 [QUALIFY] Ora UTC {hour:02d} fuori range "
                        f"[{int(self.utc_hour_min):02d}, {int(self.utc_hour_max):02d}) - scarto"
                    )
                except Exception:
                    logger.debug(
                        f"🕒 [QUALIFY] Timestamp non valido per calcolo ora UTC ({ts}) - scarto"
                    )
                return False

        # FILTRO: Weekday UTC (lista esplicita)
        if self.utc_weekday_set is not None:
            ts = spike_info.get('timestamp')
            if not self._is_in_utc_weekday_set(ts):
                try:
                    wd = self._get_utc_weekday_from_timestamp_ms(ts)
                    logger.debug(
                        f"📅 [QUALIFY] Weekday UTC {wd} non incluso in UTC_WEEKDAY={sorted(list(self.utc_weekday_set))} - scarto"
                    )
                except Exception:
                    logger.debug(
                        f"📅 [QUALIFY] Timestamp non valido per calcolo weekday UTC ({ts}) - scarto"
                    )
                return False

        # FILTRO: HL_MIN_PCT (ampiezza candela spike)
        if self.hl_min_pct is not None:
            hl_pct = spike_info.get('hl_pct')
            if hl_pct is None:
                return False
            try:
                if float(hl_pct) < float(self.hl_min_pct):
                    logger.debug(f"📏 [QUALIFY] HL% {float(hl_pct):.4f} sotto soglia {float(self.hl_min_pct):.4f} - scarto")
                    return False
            except Exception:
                return False

        # FILTRO: BODY_RATIO_MIN (corpo/escursione)
        if self.body_ratio_min is not None:
            body_ratio = spike_info.get('body_ratio')
            if body_ratio is None:
                return False
            try:
                if float(body_ratio) < float(self.body_ratio_min):
                    logger.debug(f"🕯️ [QUALIFY] BodyRatio {float(body_ratio):.6f} sotto soglia {float(self.body_ratio_min):.6f} - scarto")
                    return False
            except Exception:
                return False

        return True

    def detect_volume_spike_vectorized(self, df: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Rileva spike di volume identico al sistema live - VERSIONE VETTORIZZATA CON VERIFICHE"""
        if current_index < self.volume_lookback:
            return None

        try:
            # Estrai le candele per il lookback
            lookback_start = current_index - self.volume_lookback
            lookback_end = current_index
            lookback_candles = df.iloc[lookback_start:lookback_end]
            current_candle = df.iloc[current_index]

            # Calcola volume corrente e medio - VETTORIZZATO
            current_volume = float(current_candle['volume'])
            lookback_volumes = lookback_candles['volume'].astype(float).values

            # NUOVO: 0 = nessuna attività (evento nullo) -> non può essere spike
            if current_volume <= 0:
                logger.debug(f"📉 Volume corrente = {current_volume:.0f} (<=0) - evento nullo, skip")
                return None

            # NUOVO FILTRO: Volume corrente minimo
            if current_volume < self.min_current_volume:
                logger.debug(f"📉 Volume corrente {current_volume:.0f} sotto soglia {self.min_current_volume:.0f} - skip")
                return None

            # NUOVO FILTRO: Volume corrente massimo (inclusivo)
            if self.max_current_volume is not None and current_volume > self.max_current_volume:
                logger.debug(f"📈 Volume corrente {current_volume:.0f} sopra soglia {self.max_current_volume:.0f} - skip")
                return None

            # NUOVO: calcolo media lookback includendo gli zeri (zeri = reali)
            average_volume = float(np.mean(lookback_volumes))

            # NUOVO: se baseline è 0 (lookback tutto 0) non posso definire un ratio utile
            if average_volume <= 0:
                logger.debug("📉 Average volume = 0 nel lookback (nessuna attività) - skip")
                return None

            volume_ratio = current_volume / average_volume

            # VERIFICA DATI RAGIONEVOLI - FIX PER VALORI IMPOSSIBILI
            if volume_ratio > 100000000:  # Threshold molto alto
                logger.warning(f"⚠️  Volume ratio IMPROBABILE: {volume_ratio:.2f} per candela {current_index}")
                logger.warning(f"   Current volume: {current_volume}, Avg volume: {average_volume}")
                return None  # Salta spike improbabile

            # OTTIMIZZAZIONE: se non supera la soglia minima, esci subito senza calcoli sui prezzi
            if volume_ratio < self.volume_threshold:
                return None

            # OTTIMIZZAZIONE: se è settata la soglia MAX e la supera, esci subito senza calcoli sui prezzi
            if self.volume_threshold_max is not None and volume_ratio > self.volume_threshold_max:
                return None

            # Calcola prezzo medio lookback e prezzo corrente - VETTORIZZATO
            lookback_prices = lookback_candles['close'].astype(float).values
            current_price = float(current_candle['close'])
            average_price = np.mean(lookback_prices)

            # VERIFICA PREZZI RAGIONEVOLI
            if current_price <= 0 or average_price <= 0:
                logger.warning(f"⚠️  Prezzo negativo o zero: current={current_price}, avg={average_price}")
                return None

            price_ratio = ((current_price - average_price) / average_price) * 100

            logger.debug(f"🔍 Candela {current_index}: volume_ratio={volume_ratio:.2f}, price_ratio={price_ratio:.2f}%")

            spike_info = {
                'symbol': 'TEMP',  # Sarà sovrascritto
                'timestamp': current_candle['timestamp'],
                'datetime': current_candle.get('datetime', pd.Timestamp(current_candle['timestamp'], unit='ms')),
                'volume_ratio': volume_ratio,
                'current_volume': current_volume,
                'average_volume': average_volume,
                'current_price': current_price,
                'average_price': average_price,
                'price_ratio': price_ratio,
                'candle_index': current_index,
                'lookback_period': self.volume_lookback
            }
            logger.debug(f"🚨 SPIKE RILEVATO: ratio={volume_ratio:.2f}, price={current_price}, avg_price={average_price:.4f}")
            return spike_info

        except Exception as e:
            logger.error(f"❌ Errore rilevamento spike vettorizzato: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def calculate_price_increase_vectorized(self, df: pd.DataFrame, spike_index: int) -> Tuple[float, int, float]:
        """Calcola l'aumento di prezzo dopo lo spike - VERSIONE VETTORIZZATA"""
        start_index = spike_index + 1
        end_index = min(spike_index + 1 + self.price_increase_window, len(df))

        if end_index <= start_index:
            logger.debug(f"📏 Finestra prezzo insufficiente: spike_index={spike_index}, df_len={len(df)}")
            return 0.0, 0, 0.0

        try:
            current_price = float(df.iloc[spike_index]['close'])
            future_highs = df.iloc[start_index:end_index]['high'].astype(float).values

            if len(future_highs) == 0:
                return 0.0, 0, 0.0

            max_future_price = np.max(future_highs)
            price_increase = ((max_future_price - current_price) / current_price) * 100

            # Trova l'indice del picco
            peak_index = np.argmax(future_highs)
            lead_time = peak_index + 1  # Candele dopo lo spike

            logger.debug(f"📈 Aumento prezzo: {price_increase:.2f}% in {lead_time} candele (da {current_price} a {max_future_price})")

            return price_increase, lead_time, max_future_price

        except Exception as e:
            logger.error(f"❌ Errore calcolo aumento prezzo vettorizzato: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return 0.0, 0, 0.0

    def calculate_price_decrease_vectorized(self, df: pd.DataFrame, spike_index: int) -> Tuple[float, int, float]:
        """
        Calcola il drawdown massimo di prezzo dopo lo spike (per lo Stop Loss) - VERSIONE VETTORIZZATA.

        Restituisce:
        - price_decrease: percentuale di diminuzione massima dal prezzo di chiusura della candela di spike
                          al minimo toccato nella finestra (valore POSITIVO, es. 5 = -5%).
        - lead_time_down: numero di candele dopo lo spike in cui avviene questo minimo.
        - down_price: prezzo minimo toccato nella finestra.
        """
        start_index = spike_index + 1
        end_index = min(spike_index + 1 + self.price_increase_window, len(df))

        if end_index <= start_index:
            logger.debug(f"📏 Finestra prezzo insufficiente per drawdown: spike_index={spike_index}, df_len={len(df)}")
            return 0.0, 0, 0.0

        try:
            current_price = float(df.iloc[spike_index]['close'])
            future_lows = df.iloc[start_index:end_index]['low'].astype(float).values

            if len(future_lows) == 0:
                return 0.0, 0, 0.0

            min_future_price = np.min(future_lows)

            if current_price <= 0 or min_future_price <= 0:
                logger.warning(f"⚠️  Prezzo negativo o zero nel calcolo drawdown: current={current_price}, min={min_future_price}")
                return 0.0, 0, 0.0

            # Drawdown come percentuale POSITIVA (quanto è sceso rispetto al prezzo dello spike)
            price_decrease = ((current_price - min_future_price) / current_price) * 100

            # Trova l'indice del minimo
            down_index = np.argmin(future_lows)
            lead_time_down = down_index + 1  # Candele dopo lo spike

            logger.debug(f"📉 Decremento prezzo: {price_decrease:.2f}% in {lead_time_down} candele (da {current_price} a {min_future_price})")

            return price_decrease, lead_time_down, min_future_price

        except Exception as e:
            logger.error(f"❌ Errore calcolo decremento prezzo vettorizzato: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return 0.0, 0, 0.0

    # ============================
    # FIX B: TP/SL NUMPY
    # ============================
    def calculate_price_increase_numpy(self, np_data: Dict[str, Any], spike_index: int) -> Tuple[float, int, float]:
        """FIX B: calcola TP su array numpy (logica identica)."""
        start_index = spike_index + 1
        end_index = min(spike_index + 1 + self.price_increase_window, len(np_data['close']))

        if end_index <= start_index:
            logger.debug(f"📏 Finestra prezzo insufficiente: spike_index={spike_index}, df_len={len(np_data['close'])}")
            return 0.0, 0, 0.0

        try:
            close = np_data['close']
            high = np_data['high']

            current_price = float(close[spike_index])
            future_highs = high[start_index:end_index]

            if future_highs.size == 0:
                return 0.0, 0, 0.0

            max_future_price = float(np.max(future_highs))
            price_increase = ((max_future_price - current_price) / current_price) * 100

            peak_index = int(np.argmax(future_highs))
            lead_time = peak_index + 1

            logger.debug(f"📈 Aumento prezzo: {price_increase:.2f}% in {lead_time} candele (da {current_price} a {max_future_price})")

            return float(price_increase), int(lead_time), float(max_future_price)

        except Exception as e:
            logger.error(f"❌ Errore calcolo aumento prezzo numpy: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return 0.0, 0, 0.0

    def calculate_price_decrease_numpy(self, np_data: Dict[str, Any], spike_index: int) -> Tuple[float, int, float]:
        """FIX B: calcola SL/drawdown su array numpy (logica identica)."""
        start_index = spike_index + 1
        end_index = min(spike_index + 1 + self.price_increase_window, len(np_data['close']))

        if end_index <= start_index:
            logger.debug(f"📏 Finestra prezzo insufficiente per drawdown: spike_index={spike_index}, df_len={len(np_data['close'])}")
            return 0.0, 0, 0.0

        try:
            close = np_data['close']
            low = np_data['low']

            current_price = float(close[spike_index])
            future_lows = low[start_index:end_index]

            if future_lows.size == 0:
                return 0.0, 0, 0.0

            min_future_price = float(np.min(future_lows))

            if current_price <= 0 or min_future_price <= 0:
                logger.warning(f"⚠️  Prezzo negativo o zero nel calcolo drawdown: current={current_price}, min={min_future_price}")
                return 0.0, 0, 0.0

            price_decrease = ((current_price - min_future_price) / current_price) * 100

            down_index = int(np.argmin(future_lows))
            lead_time_down = down_index + 1

            logger.debug(f"📉 Decremento prezzo: {price_decrease:.2f}% in {lead_time_down} candele (da {current_price} a {min_future_price})")

            return float(price_decrease), int(lead_time_down), float(min_future_price)

        except Exception as e:
            logger.error(f"❌ Errore calcolo decremento prezzo numpy: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return 0.0, 0, 0.0

    def _filter_spikes_by_price_ratio(self, spikes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applica i filtri PRICE_RATIO_AT_SPIKE_MIN / MAX secondo PRICE_RATIO_AT_SPIKE_MODE:

        MODE=TAILS (DEFAULT, logica attuale):
          - MIN: tiene spike con price_ratio <= MIN (da -∞ a MIN)
          - MAX: tiene spike con price_ratio >= MAX (da MAX a +∞)
          - Se entrambi settati → condizione OR.
          - Se entrambi None → nessun filtro.

        MODE=RANGE:
          - Tiene spike con MIN <= price_ratio <= MAX (inclusivo).
          - Richiede MIN e MAX entrambi settati e MIN < MAX (validato a monte).
        """
        mode = self.price_ratio_at_spike_mode
        min_v = self.price_ratio_at_spike_min
        max_v = self.price_ratio_at_spike_max

        # Nessun filtro configurato (solo per TAILS; per RANGE la config valida MIN/MAX)
        if mode == 'TAILS' and min_v is None and max_v is None:
            return spikes

        filtered: List[Dict[str, Any]] = []

        # MODE RANGE
        if mode == 'RANGE':
            # Difensivo: in teoria già validato in __init__/load_simulator_config
            if min_v is None or max_v is None:
                return []  # non qualificare nulla se config è incoerente
            min_f = float(min_v)
            max_f = float(max_v)

            for s in spikes:
                pr = s.get('price_ratio')
                if pr is None:
                    continue  # se manca il dato, per sicurezza scarto
                try:
                    pr_f = float(pr)
                except Exception:
                    continue
                if pr_f >= min_f and pr_f <= max_f:
                    filtered.append(s)

            return filtered

        # MODE TAILS
        for s in spikes:
            pr = s.get('price_ratio')
            if pr is None:
                continue  # se manca il dato, per sicurezza scarto

            keep = False
            # Da -∞ a MIN
            if min_v is not None and pr <= min_v:
                keep = True
            # Da MAX a +∞
            if max_v is not None and pr >= max_v:
                keep = True

            if keep:
                filtered.append(s)

        return filtered

    def analyze_single_symbol(self, symbol: str, apply_price_ratio_filter: bool = True) -> List[Dict[str, Any]]:
        """
        Analizza un singolo simbolo e restituisce tutti gli spike.

        apply_price_ratio_filter:
          - True  -> applica PRICE_RATIO_AT_SPIKE_* (comportamento standard)
          - False -> NON applica i filtri price_ratio (usato dalla modalità CORE per l'optimizer)
        """
        logger.info(f"🔍 Analisi simbolo: {symbol}")

        try:
            spikes: List[Dict[str, Any]] = []

            # FIX B/MEMMAP: carica direttamente np_data (memmap se presente)
            np_data = self.load_symbol_numpy_data(symbol)

            df_len = int(len(np_data['ts']))

            # Scansiona tutte le candele (tranne le ultime per avere finestra prezzo)
            max_index = df_len - self.price_increase_window - 1
            logger.debug(f"📊 Simbolo {symbol}: {df_len} candele, analisi da {self.volume_lookback} a {max_index}")

            if max_index <= self.volume_lookback:
                logger.debug(
                    f"📏 Simbolo {symbol}: finestra non sufficiente per analisi "
                    f"(max_index={max_index}, lookback={self.volume_lookback})"
                )
                return []

            spikes_found = 0

            # ============================
            # FIX 7 (B): CORE spike detection vettoriale (candidati) + cooldown su candidati
            # Ottimizzazione: calcola price_ratio SOLO sui candidati che superano volume_threshold,
            # replicando la logica "calcola prezzi solo se supera soglia volume"
            # ============================
            lb = self.volume_lookback

            ts = np_data['ts']
            vol = np_data['vol']
            close = np_data['close']
            high = np_data['high']
            low = np_data['low']
            open_ = np_data['open']
            cvol = np_data['cvol']
            cclose = np_data['cclose']

            # Indici candidabili: [lb, max_index)
            t_build = time.perf_counter()
            idx_all = np.arange(lb, max_index, dtype=np.int64)

            # Volumi: current e media lookback [i-lb, i)
            curr_vol_all = vol[idx_all]
            avg_vol_all = (cvol[idx_all] - cvol[idx_all - lb]) / float(lb)

            # Sanity core (identico a detect_volume_spike_core_numpy):
            # - current_volume > 0
            # - average_volume > 0
            base_ok = (curr_vol_all > 0) & (avg_vol_all > 0)

            # volume_ratio (safe divide)
            volume_ratio_all = np.zeros_like(curr_vol_all, dtype=float)
            np.divide(curr_vol_all, avg_vol_all, out=volume_ratio_all, where=avg_vol_all > 0)

            # sanity ratio massimo + threshold minimo
            ratio_ok = volume_ratio_all <= 100000000.0
            thr_ok = volume_ratio_all >= float(self.volume_threshold)

            pre_mask = base_ok & ratio_ok & thr_ok
            if not np.any(pre_mask):
                logger.info(f"📈 Simbolo {symbol} completato: 0 spike trovati (prima dei filtri price_ratio_at_spike)")
                return []

            cand_idx = idx_all[pre_mask]
            cand_vr = volume_ratio_all[pre_mask]
            cand_cv = curr_vol_all[pre_mask]
            cand_av = avg_vol_all[pre_mask]

            # Prezzi: calcolati SOLO per i candidati (come logica originale)
            cand_cp = close[cand_idx]
            cand_ap = (cclose[cand_idx] - cclose[cand_idx - lb]) / float(lb)

            price_ok = (cand_cp > 0) & (cand_ap > 0)
            if not np.any(price_ok):
                logger.info(f"📈 Simbolo {symbol} completato: 0 spike trovati (prima dei filtri price_ratio_at_spike)")
                return []

            cand_idx = cand_idx[price_ok]
            cand_vr = cand_vr[price_ok]
            cand_cv = cand_cv[price_ok]
            cand_av = cand_av[price_ok]
            cand_cp = cand_cp[price_ok]
            cand_ap = cand_ap[price_ok]

            # price_ratio (%)
            cand_pr = ((cand_cp - cand_ap) / cand_ap) * 100.0

            # candle stats: HL% e body_ratio sulla candela spike
            cand_high = high[cand_idx]
            cand_low = low[cand_idx]
            cand_open = open_[cand_idx]

            hl_range = (cand_high - cand_low)
            hl_pct = np.zeros_like(cand_cp, dtype=float)
            np.divide(hl_range, cand_cp, out=hl_pct, where=cand_cp > 0)
            hl_pct = hl_pct * 100.0

            body = np.abs(cand_cp - cand_open)
            body_ratio = np.zeros_like(cand_cp, dtype=float)
            np.divide(body, hl_range, out=body_ratio, where=hl_range > 0)

            logger.debug(
                f"🔎 {symbol}: candidati CORE trovati={cand_idx.size} "
                f"(range i=[{lb}, {max_index}))"
            )

            # gestione cooldown (sui candidati)
            next_allowed_index = -1

            t_loop = time.perf_counter()
            for j in range(cand_idx.size):
                i = int(cand_idx[j])

                # Applica cooldown: salta candele finché non scade il cooldown
                if self.spike_cooldown_candles > 0 and i < next_allowed_index:
                    continue

                # Cooldown scatta SEMPRE sul core spike (indipendente dai filtri, inclusa ratio_max)
                if self.spike_cooldown_candles > 0:
                    next_allowed_index = i + int(self.spike_cooldown_candles)

                ts_i = ts[i]
                core_spike = {
                    'symbol': 'TEMP',  # Sarà sovrascritto
                    'timestamp': ts_i,
                    'datetime': pd.Timestamp(ts_i, unit='ms'),
                    'volume_ratio': float(cand_vr[j]),
                    'current_volume': float(cand_cv[j]),
                    'average_volume': float(cand_av[j]),
                    'current_price': float(cand_cp[j]),
                    'average_price': float(cand_ap[j]),
                    'price_ratio': float(cand_pr[j]),
                    'hl_pct': float(hl_pct[j]),
                    'body_ratio': float(body_ratio[j]),
                    'candle_index': i,
                    'lookback_period': lb
                }

                logger.debug(
                    f"🔍 [CORE] Candela {i}: volume_ratio={core_spike['volume_ratio']:.2f}, "
                    f"price_ratio={core_spike['price_ratio']:.2f}%"
                )

                # 3) Qualifica spike (a valle): decide se loggare/salvare l'evento
                if not self._qualify_spike_post_filters(core_spike):
                    continue

                spike_data = core_spike
                spike_data['symbol'] = symbol

                # TP: calcolo logica verso l'alto (numpy)
                price_increase, lead_time, peak_price = self.calculate_price_increase_numpy(np_data, i)

                # SL: calcolo logica verso il basso (numpy)
                price_decrease, lead_time_down, down_price = self.calculate_price_decrease_numpy(np_data, i)

                spike_data['price_increase'] = price_increase
                spike_data['lead_time'] = lead_time
                spike_data['peak_price'] = peak_price

                # Nuovi campi per Stop Loss / drawdown
                spike_data['price_decrease'] = price_decrease
                spike_data['lead_time_down'] = lead_time_down
                spike_data['down_price'] = down_price

                # LOGICA LONG/SHORT CON SOGLIA COMUNE E OPZIONE A PER PARITÀ
                thr = self.min_price_increase_percent
                up_reaches = price_increase >= thr
                down_reaches = price_decrease >= thr
                t_up = lead_time
                t_down = lead_time_down

                is_ok_long = False
                is_ok_short = False
                direction = 'NONE'

                if up_reaches and not down_reaches:
                    is_ok_long = True
                    direction = 'LONG'
                elif down_reaches and not up_reaches:
                    is_ok_short = True
                    direction = 'SHORT'
                elif up_reaches and down_reaches:
                    if t_up < t_down:
                        is_ok_long = True
                        direction = 'LONG'
                    elif t_down < t_up:
                        is_ok_short = True
                        direction = 'SHORT'
                    else:
                        # Opzione A: in caso di parità, priorità al LONG
                        is_ok_long = True
                        direction = 'LONG'
                else:
                    is_ok_long = False
                    is_ok_short = False
                    direction = 'NONE'

                spike_data['is_ok'] = is_ok_long
                spike_data['is_ok_short'] = is_ok_short
                spike_data['direction'] = direction

                spike_data['price_increase_window'] = self.price_increase_window
                spike_data['min_price_increase_required'] = self.min_price_increase_percent

                spikes.append(spike_data)
                spikes_found += 1

                status = "✅ OK_LONG" if spike_data['is_ok'] else ("✅ OK_SHORT" if spike_data['is_ok_short'] else "❌ KO")
                logger.debug(f"{status} Spike {spikes_found}: ratio={spike_data['volume_ratio']:.2f}, increase={price_increase:.2f}%, decrease={price_decrease:.2f}%")
                logger.debug(f"   Drawdown: {price_decrease:.2f}% in {lead_time_down} candele (down_price={down_price}), direction={direction}")

            logger.info(f"📈 Simbolo {symbol} completato: {spikes_found} spike trovati (prima dei filtri price_ratio_at_spike)")

            # 🔎 Filtro per price_ratio_at_spike (post-produzione)
            if apply_price_ratio_filter:
                before_filter_count = len(spikes)
                spikes = self._filter_spikes_by_price_ratio(spikes)
                after_filter_count = len(spikes)

                if before_filter_count != after_filter_count:
                    logger.info(
                        f"🔎 {symbol}: filtri price_ratio_at_spike applicati - "
                        f"prima={before_filter_count}, dopo={after_filter_count}, "
                        f"MODE={self.price_ratio_at_spike_mode}, MIN={self.price_ratio_at_spike_min}, MAX={self.price_ratio_at_spike_max}"
                    )

            return spikes

        except Exception as e:
            logger.error(f"❌ Errore analisi {symbol}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []

    # ================================
    #  LIVELLO B: CORE ARRAYS (PER OPTIMIZER)
    # ================================
    def _precompute_future_window_extrema(self, high: np.ndarray, low: np.ndarray, window: int) -> Dict[str, np.ndarray]:
        """
        Precompute O(n) di max(high) e min(low) nelle finestre forward:
          future window per indice i -> [i+1, i+window]
        Restituisce anche i lead-time (1..window) verso il primo estremo.
        """
        n = int(len(high))
        if n == 0 or window <= 0:
            return {
                "max_future_high": np.zeros(n, dtype=np.float64),
                "lead_time_up": np.zeros(n, dtype=np.int32),
                "min_future_low": np.zeros(n, dtype=np.float64),
                "lead_time_down": np.zeros(n, dtype=np.int32),
            }

        max_future_high = np.zeros(n, dtype=np.float64)
        lead_time_up = np.zeros(n, dtype=np.int32)
        min_future_low = np.zeros(n, dtype=np.float64)
        lead_time_down = np.zeros(n, dtype=np.int32)

        dq_max = deque()
        dq_min = deque()

        for i in range(n - 1, -1, -1):
            left = i + 1
            right = i + window

            # rimuovi indici fuori finestra
            while dq_max and dq_max[0] > right:
                dq_max.popleft()
            while dq_min and dq_min[0] > right:
                dq_min.popleft()

            # aggiungi nuovo indice left (se valido)
            if left < n:
                while dq_max and float(high[left]) > float(high[dq_max[-1]]):
                    dq_max.pop()
                dq_max.append(left)

                while dq_min and float(low[left]) < float(low[dq_min[-1]]):
                    dq_min.pop()
                dq_min.append(left)

            # garantisci lower bound
            while dq_max and dq_max[0] < left:
                dq_max.popleft()
            while dq_min and dq_min[0] < left:
                dq_min.popleft()

            if dq_max:
                jmax = dq_max[0]
                max_future_high[i] = float(high[jmax])
                lead_time_up[i] = int(jmax - i)

            if dq_min:
                jmin = dq_min[0]
                min_future_low[i] = float(low[jmin])
                lead_time_down[i] = int(jmin - i)

        return {
            "max_future_high": max_future_high,
            "lead_time_up": lead_time_up,
            "min_future_low": min_future_low,
            "lead_time_down": lead_time_down,
        }

    def _dt_strings_from_ts(self, ts_arr: np.ndarray, profile: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Converte un array di timestamp ms UTC in stringhe datetime usando cache interna.
        """
        if ts_arr.size == 0:
            return np.array([], dtype=object)

        t_dt = time.perf_counter()
        out_dt = np.empty(ts_arr.size, dtype=object)
        unique_ts, inverse = np.unique(ts_arr, return_inverse=True)
        unique_dt = np.empty(unique_ts.size, dtype=object)

        for k, ts_i in enumerate(unique_ts):
            ts_key = int(ts_i)
            dt_str = self._dt_utc_cache.get(ts_key)
            if dt_str is None:
                dt_str = datetime.utcfromtimestamp(ts_key / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                if len(self._dt_utc_cache) >= 200000:
                    self._dt_utc_cache.clear()
                self._dt_utc_cache[ts_key] = dt_str
            unique_dt[k] = dt_str

        out_dt[:] = unique_dt[inverse]
        if profile is not None:
            profile["datetime_s"] += (time.perf_counter() - t_dt)
        return out_dt

    def analyze_single_symbol_core_arrays(self, symbol: str) -> Dict[str, Any]:
        """
        Analizza un singolo simbolo in modalità CORE (senza filtri price_ratio_at_spike)
        e ritorna ARRAYS compatti per l'optimizer (Livello B).
        """
        logger.debug(f"🔍 [CORE] Analisi simbolo (arrays): {symbol}")

        core_subprofile = bool(self.config.get("core_subprofile", False))
        prof = {
            "load_np_s": 0.0,
            "extrema_s": 0.0,
            "candidate_build_s": 0.0,
            "candidate_loop_s": 0.0,
            "post_filter_s": 0.0,
            "price_prep_s": 0.0,
            "price_eval_s": 0.0,
            "datetime_s": 0.0,
        }

        try:
            t_load = time.perf_counter()
            np_data = self.load_symbol_numpy_data(symbol)
            if core_subprofile:
                prof["load_np_s"] += (time.perf_counter() - t_load)

            df_len = int(len(np_data['ts']))
            max_index = df_len - self.price_increase_window - 1

            if max_index <= self.volume_lookback:
                return {
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
                }

            lb = self.volume_lookback

            ts = np_data['ts']
            vol = np_data['vol']
            close = np_data['close']
            high = np_data['high']
            low = np_data['low']
            open_ = np_data['open']
            cvol = np_data['cvol']
            cclose = np_data['cclose']

            use_precompute_extrema = (self.core_extrema_mode == 'precompute')
            max_future_high = None
            lead_time_up = None
            min_future_low = None
            lead_time_down_arr = None
            eval_window = int(self.price_increase_window)
            if use_precompute_extrema:
                t_extrema = time.perf_counter()
                extrema = self._precompute_future_window_extrema(high, low, eval_window)
                if core_subprofile:
                    prof["extrema_s"] += (time.perf_counter() - t_extrema)
                max_future_high = extrema['max_future_high']
                lead_time_up = extrema['lead_time_up']
                min_future_low = extrema['min_future_low']
                lead_time_down_arr = extrema['lead_time_down']

            use_post_filters = (
                float(self.min_current_volume) > 0.0
                or self.max_current_volume is not None
                or self.volume_threshold_max is not None
                or (self.utc_hour_min is not None and self.utc_hour_max is not None)
                or self.utc_weekday_set is not None
                or self.hl_min_pct is not None
                or self.body_ratio_min is not None
            )

            t_build = time.perf_counter()
            cooldown_step = int(self.spike_cooldown_candles) if self.spike_cooldown_candles > 0 else 0
            if self.core_use_numba and NUMBA_AVAILABLE:
                cand_idx, cand_vr, cand_cv, cand_ap, cand_pr, hl_pct, body_ratio = _build_candidates_numba(
                    vol,
                    cvol,
                    close,
                    cclose,
                    open_,
                    high,
                    low,
                    int(lb),
                    int(max_index),
                    float(self.volume_threshold),
                    int(cooldown_step),
                )
                cand_av = None
                if use_post_filters:
                    if cand_idx.size > 0:
                        cand_av = (cvol[cand_idx] - cvol[cand_idx - lb]) / float(lb)
                    else:
                        cand_av = np.empty(0, dtype=np.float64)
                cand_cp = close[cand_idx]
            else:
                idx_all = np.arange(lb, max_index, dtype=np.int64)
                curr_vol_all = vol[idx_all]
                avg_vol_all = (cvol[idx_all] - cvol[idx_all - lb]) / float(lb)

                base_ok = (curr_vol_all > 0) & (avg_vol_all > 0)

                volume_ratio_all = np.zeros_like(curr_vol_all, dtype=float)
                np.divide(curr_vol_all, avg_vol_all, out=volume_ratio_all, where=avg_vol_all > 0)

                ratio_ok = volume_ratio_all <= 100000000.0
                thr_ok = volume_ratio_all >= float(self.volume_threshold)

                pre_mask = base_ok & ratio_ok & thr_ok
                if not np.any(pre_mask):
                    return {
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
                    }

                cand_idx = idx_all[pre_mask]
                cand_vr = volume_ratio_all[pre_mask]
                cand_cv = curr_vol_all[pre_mask]
                cand_av = avg_vol_all[pre_mask] if use_post_filters else None

                cand_cp = close[cand_idx]
                cand_ap = (cclose[cand_idx] - cclose[cand_idx - lb]) / float(lb)

                price_ok = (cand_cp > 0) & (cand_ap > 0)
                if not np.any(price_ok):
                    return {
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
                    }

                cand_idx = cand_idx[price_ok]
                cand_vr = cand_vr[price_ok]
                cand_cv = cand_cv[price_ok]
                if cand_av is not None:
                    cand_av = cand_av[price_ok]
                cand_cp = cand_cp[price_ok]
                cand_ap = cand_ap[price_ok]

                cand_pr = ((cand_cp - cand_ap) / cand_ap) * 100.0

                cand_high = high[cand_idx]
                cand_low = low[cand_idx]
                cand_open = open_[cand_idx]

                hl_range = (cand_high - cand_low)
                hl_pct = np.zeros_like(cand_cp, dtype=float)
                np.divide(hl_range, cand_cp, out=hl_pct, where=cand_cp > 0)
                hl_pct = hl_pct * 100.0

                body = np.abs(cand_cp - cand_open)
                body_ratio = np.zeros_like(cand_cp, dtype=float)
                np.divide(body, hl_range, out=body_ratio, where=hl_range > 0)

            if cand_idx.size == 0:
                return {
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
                }

            if cooldown_step > 0 and cand_idx.size > 1 and not (self.core_use_numba and NUMBA_AVAILABLE):
                keep_pos = []
                j_keep = 0
                while j_keep < cand_idx.size:
                    keep_pos.append(j_keep)
                    next_allowed = int(cand_idx[j_keep]) + cooldown_step
                    j_keep = int(np.searchsorted(cand_idx, next_allowed, side='left'))

                keep_pos = np.asarray(keep_pos, dtype=np.int64)
                cand_idx = cand_idx[keep_pos]
                cand_vr = cand_vr[keep_pos]
                cand_cv = cand_cv[keep_pos]
                if cand_av is not None:
                    cand_av = cand_av[keep_pos]
                cand_cp = cand_cp[keep_pos]
                cand_ap = cand_ap[keep_pos]

            if cand_idx.size == 0:
                return {
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
                }

            if core_subprofile:
                prof["candidate_build_s"] += (time.perf_counter() - t_build)

            cand_max_future_price = None
            cand_min_future_price = None
            cand_lead_time = None
            cand_lead_time_down = None
            if not use_precompute_extrema and cand_idx.size > 0 and eval_window > 0:
                t_prep = time.perf_counter()
                if self.core_use_numba and NUMBA_AVAILABLE:
                    (
                        cand_max_future_price,
                        cand_lead_time,
                        cand_min_future_price,
                        cand_lead_time_down,
                    ) = _compute_candidate_future_extrema_numba(high, low, cand_idx, int(eval_window))
                else:
                    (
                        cand_max_future_price,
                        cand_lead_time,
                        cand_min_future_price,
                        cand_lead_time_down,
                    ) = _compute_candidate_future_extrema_numpy(high, low, cand_idx, int(eval_window))
                if core_subprofile:
                    prof["price_prep_s"] += (time.perf_counter() - t_prep)

            t_loop = time.perf_counter()

            if use_precompute_extrema:
                lead_time_arr = lead_time_up[cand_idx].astype(np.int32, copy=False)
                lead_time_down_vec = lead_time_down_arr[cand_idx].astype(np.int32, copy=False)
                peak_price_arr = max_future_high[cand_idx].astype(np.float64, copy=False)
                down_price_arr = min_future_low[cand_idx].astype(np.float64, copy=False)
            else:
                lead_time_arr = cand_lead_time.astype(np.int32, copy=False) if cand_lead_time is not None else np.zeros(cand_idx.size, dtype=np.int32)
                lead_time_down_vec = cand_lead_time_down.astype(np.int32, copy=False) if cand_lead_time_down is not None else np.zeros(cand_idx.size, dtype=np.int32)
                peak_price_arr = cand_max_future_price.astype(np.float64, copy=False) if cand_max_future_price is not None else np.zeros(cand_idx.size, dtype=np.float64)
                down_price_arr = cand_min_future_price.astype(np.float64, copy=False) if cand_min_future_price is not None else np.zeros(cand_idx.size, dtype=np.float64)

            t_price = time.perf_counter()
            price_increase_arr = np.zeros(cand_idx.size, dtype=np.float64)
            price_decrease_arr = np.zeros(cand_idx.size, dtype=np.float64)

            valid_up = (cand_cp > 0.0) & (peak_price_arr > 0.0) & (lead_time_arr > 0)
            valid_down = (cand_cp > 0.0) & (down_price_arr > 0.0) & (lead_time_down_vec > 0)

            np.divide(
                (peak_price_arr - cand_cp) * 100.0,
                cand_cp,
                out=price_increase_arr,
                where=valid_up,
            )
            np.divide(
                (cand_cp - down_price_arr) * 100.0,
                cand_cp,
                out=price_decrease_arr,
                where=valid_down,
            )

            lead_time_eff = lead_time_arr.copy()
            lead_time_down_eff = lead_time_down_vec.copy()
            lead_time_eff[~valid_up] = 0
            lead_time_down_eff[~valid_down] = 0

            if core_subprofile:
                prof["price_eval_s"] += (time.perf_counter() - t_price)

            thr = float(self.min_price_increase_percent)
            up_reaches = price_increase_arr >= thr
            down_reaches = price_decrease_arr >= thr

            is_ok_long_arr = up_reaches & ((~down_reaches) | (lead_time_eff <= lead_time_down_eff))
            is_ok_short_arr = down_reaches & ((~up_reaches) | (lead_time_down_eff < lead_time_eff))

            if use_post_filters:
                keep_mask = np.zeros(cand_idx.size, dtype=bool)
                for j in range(cand_idx.size):
                    core_spike = {
                        'symbol': symbol,
                        'timestamp': int(ts[cand_idx[j]]),
                        'volume_ratio': float(cand_vr[j]),
                        'current_volume': float(cand_cv[j]),
                        'average_volume': float(cand_av[j]),
                        'price_ratio': float(cand_pr[j]),
                        'hl_pct': float(hl_pct[j]),
                        'body_ratio': float(body_ratio[j]),
                    }
                    t_post = time.perf_counter()
                    ok_post = self._qualify_spike_post_filters(core_spike)
                    if core_subprofile:
                        prof["post_filter_s"] += (time.perf_counter() - t_post)
                    if ok_post:
                        keep_mask[j] = True
            else:
                keep_mask = None

            if core_subprofile:
                prof["candidate_loop_s"] += (time.perf_counter() - t_loop)

            if keep_mask is not None and not np.any(keep_mask):
                return {
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
                }

            if keep_mask is not None:
                out_ts_arr = ts[cand_idx][keep_mask].astype(np.int64, copy=False)
                out_curr_vol = cand_cv[keep_mask].astype(np.float64, copy=False)
                out_vol_ratio = cand_vr[keep_mask].astype(np.float64, copy=False)
                out_price_ratio = cand_pr[keep_mask].astype(np.float64, copy=False)
                out_hl_pct = hl_pct[keep_mask].astype(np.float64, copy=False)
                out_body_ratio = body_ratio[keep_mask].astype(np.float64, copy=False)
                out_is_ok = is_ok_long_arr[keep_mask].astype(bool, copy=False)
                out_is_ok_short = is_ok_short_arr[keep_mask].astype(bool, copy=False)
                out_price_increase = price_increase_arr[keep_mask].astype(np.float64, copy=False)
                out_lead_time = lead_time_eff[keep_mask].astype(np.int32, copy=False)
                out_price_decrease = price_decrease_arr[keep_mask].astype(np.float64, copy=False)
                out_lead_time_down = lead_time_down_eff[keep_mask].astype(np.int32, copy=False)
            else:
                out_ts_arr = ts[cand_idx].astype(np.int64, copy=False)
                out_curr_vol = cand_cv.astype(np.float64, copy=False)
                out_vol_ratio = cand_vr.astype(np.float64, copy=False)
                out_price_ratio = cand_pr.astype(np.float64, copy=False)
                out_hl_pct = hl_pct.astype(np.float64, copy=False)
                out_body_ratio = body_ratio.astype(np.float64, copy=False)
                out_is_ok = is_ok_long_arr.astype(bool, copy=False)
                out_is_ok_short = is_ok_short_arr.astype(bool, copy=False)
                out_price_increase = price_increase_arr.astype(np.float64, copy=False)
                out_lead_time = lead_time_eff.astype(np.int32, copy=False)
                out_price_decrease = price_decrease_arr.astype(np.float64, copy=False)
                out_lead_time_down = lead_time_down_eff.astype(np.int32, copy=False)

            include_dt_strings = bool(self.config.get("core_include_dt_strings", True))
            if include_dt_strings:
                out_dt_arr = self._dt_strings_from_ts(out_ts_arr, prof if core_subprofile else None)
            else:
                out_dt_arr = np.array([], dtype=object)
            out_payload = {
                "symbol": np.full(out_ts_arr.size, symbol, dtype=object),
                "ts": out_ts_arr,
                "dt_str": out_dt_arr,
                "curr_vol": out_curr_vol,
                "vol_ratio": out_vol_ratio,
                "price_ratio": out_price_ratio,
                "hl_pct": out_hl_pct,
                "body_ratio": out_body_ratio,
                "is_ok": out_is_ok,
                "is_ok_short": out_is_ok_short,
                "price_increase": out_price_increase,
                "lead_time": out_lead_time,
                "price_decrease": out_price_decrease,
                "lead_time_down": out_lead_time_down,
            }
            if core_subprofile:
                out_payload["__profile"] = dict(prof)
            return out_payload

        except Exception as e:
            logger.error(f"❌ [CORE] Errore analisi arrays {symbol}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return {
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
            }

    def process_symbol_batch(self, symbol_batch: List[str]) -> List[Dict[str, Any]]:
        """Processa un batch di simboli - per multiprocessing"""
        batch_results = []
        logger.debug(f"🔄 Inizio processing batch con {len(symbol_batch)} simboli")

        for symbol in symbol_batch:
            try:
                spikes = self.analyze_single_symbol(symbol)
                batch_results.extend(spikes)

                # Aggiorna statistiche
                self.symbols_processed += 1
                elapsed = time.time() - self.start_time
                progress_pct = (self.symbols_processed / self.total_symbols) * 100

                # Memory monitoring
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                logger.debug(f"✅ Batch progress: {self.symbols_processed}/{self.total_symbols} ({progress_pct:.1f}%) - Memoria: {memory_usage:.1f}MB")

            except Exception as e:
                logger.error(f"❌ Errore batch {symbol}: {e}")
                self.symbols_processed += 1
                continue

        logger.debug(f"✅ Batch completato: {len(batch_results)} spike totali")
        return batch_results

    def analyze_all_symbols_parallel(self) -> List[Dict[str, Any]]:
        """Analizza tutti i simboli in parallelo - ULTRA VELOCE (con fallback sequenziale in processi daemonic o forzati)."""
        symbols = self.get_available_symbols()
        all_spikes: List[Dict[str, Any]] = []

        self.start_time = time.time()
        self.symbols_processed = 0

        # Se il processo corrente è "daemonic" (es. worker di multiprocessing.Pool),
        # NON è permesso creare ulteriori processi figli.
        # In questo caso usiamo una scansione SEQUENZIALE sui simboli.
        current_proc = mp.current_process()
        is_daemonic = getattr(current_proc, "daemon", False)

        # Usiamo multiprocessing SOLO se:
        # - max_workers > 1
        # - il processo NON è daemonic
        # - non è abilitato il force_sequential
        use_multiprocessing = (
            self.max_workers is not None
            and self.max_workers > 1
            and not is_daemonic
            and not self.force_sequential
        )

        logger.info(f"🚀 Avvio analisi simboli: {len(symbols)} totali")
        logger.info(
            f"⚙️  Configurazione: max_workers={self.max_workers}, batch size={self.batch_size}, "
            f"use_multiprocessing={use_multiprocessing}, is_daemonic={is_daemonic}, force_sequential={self.force_sequential}"
        )

        # Memory info iniziale
        memory_info = psutil.virtual_memory()
        logger.info(f"💾 Memoria disponibile: {memory_info.available / 1024 / 1024:.0f}MB ({memory_info.percent}% utilizzata)")

        # Se non usiamo multiprocessing, processiamo i simboli in modo sequenziale
        if not use_multiprocessing:
            logger.info("🔧 Esecuzione SEQUENZIALE dei simboli (no ProcessPoolExecutor).")
            for idx, symbol in enumerate(symbols, start=1):
                try:
                    spikes = self.analyze_single_symbol(symbol)
                    all_spikes.extend(spikes)

                    self.symbols_processed = idx
                    elapsed = time.time() - self.start_time
                    progress_pct = (self.symbols_processed / self.total_symbols) * 100 if self.total_symbols else 0.0
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                    logger.info(
                        f"✅ Simbolo {idx}/{len(symbols)} elaborato "
                        f"({progress_pct:.1f}% completato) - Spike cumulati: {len(all_spikes)} - "
                        f"Memoria: {memory_usage:.1f}MB - Tempo trascorso: {elapsed:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"❌ Errore analisi simbolo {symbol} in modalità sequenziale: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    continue

            total_time = time.time() - self.start_time if self.start_time else 0.0
            if total_time > 0 and self.total_symbols:
                logger.info(f"🎯 Analisi completata (sequenziale): {len(all_spikes)} spike in {total_time:.1f}s")
                logger.info(f"⚡ Performance: {self.total_symbols / total_time:.2f} simboli/secondo")
            else:
                logger.info("🎯 Analisi completata (sequenziale): nessun simbolo processato.")
            return all_spikes

        # --- Modalità MULTIPROCESSING con ProcessPoolExecutor (come in origine) ---
        logger.info(f"🚀 Avvio analisi parallela di {len(symbols)} simboli")
        logger.info(f"⚙️  Configurazione multiprocessing: {self.max_workers} workers, batch size {self.batch_size}")

        # Suddividi in batch
        batches = [symbols[i:i + self.batch_size] for i in range(0, len(symbols), self.batch_size)]
        total_batches = len(batches)

        logger.info(f"📦 Suddivisione in {total_batches} batch")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_symbol_batch, batch): batch_num
                for batch_num, batch in enumerate(batches)
            }

            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_start = time.time()
                    batch_results = future.result()
                    batch_time = time.time() - batch_start

                    all_spikes.extend(batch_results)

                    elapsed_total = time.time() - self.start_time
                    spikes_so_far = len(all_spikes)
                    batches_done = batch_num + 1

                    # Calcola ETA
                    time_per_batch = elapsed_total / batches_done if batches_done > 0 else 0.0
                    remaining_batches = total_batches - batches_done
                    eta_seconds = time_per_batch * remaining_batches
                    eta_str = str(datetime.utcfromtimestamp(eta_seconds).strftime('%H:%M:%S'))

                    # Memory monitoring
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

                    logger.info(f"✅ Batch {batches_done}/{total_batches} completato in {batch_time:.1f}s")
                    logger.info(
                        f"📊 Progresso: {self.symbols_processed}/{self.total_symbols} simboli - "
                        f"{spikes_so_far} spike - ETA: {eta_str}"
                    )
                    logger.info(f"💾 Memoria processo: {memory_usage:.1f}MB")

                except Exception as e:
                    logger.error(f"❌ Errore batch {batch_num}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")

        total_time = time.time() - self.start_time if self.start_time else 0.0
        if total_time > 0 and self.total_symbols:
            logger.info(f"🎯 Analisi completata: {len(all_spikes)} spike in {total_time:.1f}s")
            logger.info(f"⚡ Performance: {self.total_symbols / total_time:.2f} simboli/secondo")
        else:
            logger.info("🎯 Analisi completata: nessun simbolo processato.")

        return all_spikes

    def _weekday_to_italian_name(self, weekday: int) -> str:
        mapping = {
            0: "Lunedì",
            1: "Martedì",
            2: "Mercoledì",
            3: "Giovedì",
            4: "Venerdì",
            5: "Sabato",
            6: "Domenica",
        }
        return mapping.get(weekday, str(weekday))

    def _calculate_temporal_stats(self, spikes: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_hour: Dict[int, int] = {h: 0 for h in range(24)}
        by_weekday: Dict[int, int] = {d: 0 for d in range(7)}
        by_month: Dict[int, int] = {m: 0 for m in range(1, 13)}

        for s in spikes:
            dt = s.get('datetime')
            if dt is None:
                continue
            dt_ts = pd.to_datetime(dt)
            by_hour[dt_ts.hour] += 1
            by_weekday[dt_ts.weekday()] += 1
            by_month[dt_ts.month] += 1

        best_hour = max(by_hour.items(), key=lambda x: x[1])[0] if by_hour else None
        best_weekday = max(by_weekday.items(), key=lambda x: x[1])[0] if by_weekday else None
        best_month = max(by_month.items(), key=lambda x: x[1])[0] if by_month else None

        return {
            'by_hour': by_hour,
            'by_weekday': {self._weekday_to_italian_name(k): v for k, v in by_weekday.items()},
            'by_month': by_month,
            'best_hour': best_hour,
            'best_weekday_index': best_weekday,
            'best_weekday_name': self._weekday_to_italian_name(best_weekday) if best_weekday is not None else None,
            'best_month': best_month,
        }

    def calculate_metrics(self, all_spikes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcola tutte le metriche richieste"""
        logger.info("📐 Calcolo metriche dettagliate...")

        if not all_spikes:
            logger.warning("⚠️  Nessuno spike trovato per il calcolo metriche")
            return self.get_empty_metrics()

        # Separazione eventi OK/KO LONG
        ok_spikes = [s for s in all_spikes if s.get('is_ok')]
        ko_spikes = [s for s in all_spikes if not s.get('is_ok')]

        # Separazione eventi OK SHORT
        ok_spikes_short = [s for s in all_spikes if s.get('is_ok_short')]
        ko_spikes_short = [s for s in all_spikes if not s.get('is_ok_short')]

        logger.info(f"📊 Statistiche spike LONG: {len(ok_spikes)} OK, {len(ko_spikes)} KO")
        logger.info(f"📊 Statistiche spike SHORT: {len(ok_spikes_short)} OK, {len(ko_spikes_short)} KO")

        # Calcolo metriche base LONG
        total_spikes = len(all_spikes)
        ok_count = len(ok_spikes)
        ko_count = len(ko_spikes)
        efficiency = ok_count / total_spikes if total_spikes > 0 else 0

        # Aumento prezzo medio per OK LONG
        price_increases_ok = [s['price_increase'] for s in ok_spikes]
        avg_price_increase = np.mean(price_increases_ok) if price_increases_ok else 0

        # Lead time LONG
        lead_times = [s['lead_time'] for s in ok_spikes if s['lead_time'] > 0]
        avg_lead_time = np.mean(lead_times) if lead_times else 0
        std_lead_time = np.std(lead_times) if lead_times else 0

        # Spike strength (comune)
        spike_strengths = [s['volume_ratio'] for s in all_spikes]
        avg_spike_strength = np.mean(spike_strengths) if spike_strengths else 0

        # Prezzo in aumento al momento dello spike (comune)
        price_increasing = [s for s in all_spikes if s['price_ratio'] > 0]
        percent_price_increasing = len(price_increasing) / total_spikes if total_spikes > 0 else 0

        # Distribuzione range aumento prezzo (solo OK LONG)
        price_ranges = self.calculate_price_ranges(ok_spikes)

        # Calcolo scores LONG
        robustness_score = self.calculate_robustness_score(std_lead_time)
        composite_score = self.calculate_composite_score(
            efficiency, avg_lead_time, avg_spike_strength
        )
        final_score = composite_score * 0.7 + robustness_score * 0.3

        logger.info(f"📈 Metriche LONG calcolate: efficienza={efficiency:.3f}, lead_time={avg_lead_time:.2f}")

        # Metriche SHORT (simmetriche ma su price_decrease / lead_time_down)
        ok_count_short = len(ok_spikes_short)
        efficiency_short = ok_count_short / total_spikes if total_spikes > 0 else 0

        price_decreases_ok = [s['price_decrease'] for s in ok_spikes_short]
        avg_price_decrease = np.mean(price_decreases_ok) if price_decreases_ok else 0

        lead_times_down = [s['lead_time_down'] for s in ok_spikes_short if s.get('lead_time_down', 0) > 0]
        avg_lead_time_down = np.mean(lead_times_down) if lead_times_down else 0
        std_lead_time_down = np.std(lead_times_down) if lead_times_down else 0

        robustness_score_short = self.calculate_robustness_score(std_lead_time_down)
        composite_score_short = self.calculate_composite_score(
            efficiency_short, avg_lead_time_down, avg_spike_strength
        )
        final_score_short = composite_score_short * 0.7 + robustness_score_short * 0.3

        price_ranges_short = self.calculate_price_ranges_from_decrease(ok_spikes_short)

        logger.info(f"📈 Metriche SHORT calcolate: efficienza={efficiency_short:.3f}, lead_time_down={avg_lead_time_down:.2f}")

        # Statistiche temporali LONG e SHORT
        temporal_stats_long = self._calculate_temporal_stats(ok_spikes)
        temporal_stats_short = self._calculate_temporal_stats(ok_spikes_short)

        return {
            'summary': {
                'total_spikes': total_spikes,
                'ok_spikes': ok_count,
                'ko_spikes': ko_count,
                'efficiency': efficiency,
                'avg_price_increase': avg_price_increase,
                'avg_lead_time': avg_lead_time,
                'avg_spike_strength': avg_spike_strength,
                'percent_price_increasing': percent_price_increasing,
                'robustness_score': robustness_score,
                'composite_score': composite_score,
                'final_score': final_score
            },
            'summary_short': {
                'total_spikes': total_spikes,
                'ok_spikes': ok_count_short,
                'ko_spikes': total_spikes - ok_count_short,
                'efficiency': efficiency_short,
                'avg_price_decrease': avg_price_decrease,
                'avg_lead_time_down': avg_lead_time_down,
                'avg_spike_strength': avg_spike_strength,
                'robustness_score': robustness_score_short,
                'composite_score': composite_score_short,
                'final_score': final_score_short
            },
            'price_ranges': price_ranges,
            'price_ranges_short': price_ranges_short,
            'detailed_metrics': {
                'std_lead_time': std_lead_time,
                'min_lead_time': min(lead_times) if lead_times else 0,
                'max_lead_time': max(lead_times) if lead_times else 0,
                'min_spike_strength': min(spike_strengths) if spike_strengths else 0,
                'max_spike_strength': max(spike_strengths) if spike_strengths else 0
            },
            'detailed_metrics_short': {
                'std_lead_time_down': std_lead_time_down,
                'min_lead_time_down': min(lead_times_down) if lead_times_down else 0,
                'max_lead_time_down': max(lead_times_down) if lead_times_down else 0
            },
            'temporal_stats_long': temporal_stats_long,
            'temporal_stats_short': temporal_stats_short
        }

    def calculate_price_ranges(self, ok_spikes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calcola distribuzione eventi OK per range di aumento prezzo"""
        logger.debug("📐 Calcolo distribuzione range prezzo (LONG)...")

        ranges = {
            '0-10%': 0,
            '10-20%': 0,
            '20-30%': 0,
            '30-50%': 0,
            '50-100%': 0,
            '>100%': 0
        }

        for spike in ok_spikes:
            increase = spike['price_increase']

            if increase < 10:
                ranges['0-10%'] += 1
            elif increase < 20:
                ranges['10-20%'] += 1
            elif increase < 30:
                ranges['20-30%'] += 1
            elif increase < 50:
                ranges['30-50%'] += 1
            elif increase < 100:
                ranges['50-100%'] += 1
            else:
                ranges['>100%'] += 1

        logger.debug(f"📊 Distribuzione range LONG: {ranges}")
        return ranges

    def calculate_price_ranges_from_decrease(self, ok_spikes_short: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calcola distribuzione eventi OK SHORT per range di diminuzione prezzo"""
        logger.debug("📐 Calcolo distribuzione range prezzo (SHORT)...")

        ranges = {
            '0-10%': 0,
            '10-20%': 0,
            '20-30%': 0,
            '30-50%': 0,
            '50-100%': 0,
            '>100%': 0
        }

        for spike in ok_spikes_short:
            decrease = spike['price_decrease']

            if decrease < 10:
                ranges['0-10%'] += 1
            elif decrease < 20:
                ranges['10-20%'] += 1
            elif decrease < 30:
                ranges['20-30%'] += 1
            elif decrease < 50:
                ranges['30-50%'] += 1
            elif decrease < 100:
                ranges['50-100%'] += 1
            else:
                ranges['>100%'] += 1

        logger.debug(f"📊 Distribuzione range SHORT: {ranges}")
        return ranges

    def calculate_robustness_score(self, std_lead_time: float) -> float:
        """Calcola robustness score come nel backtest"""
        max_acceptable_std = self.price_increase_window / 2

        if std_lead_time <= 0:
            return 1.0

        robustness = 1 - (std_lead_time / max_acceptable_std)
        score = max(0, min(robustness, 1))
        logger.debug(f"🛡️  Robustness score: {score} (std_lead_time={std_lead_time:.2f})")
        return score

    def calculate_composite_score(self, efficiency: float, avg_lead_time: float, avg_spike_strength: float) -> float:
        """Calcola composite score come nel backtest"""
        # Normalizza lead time
        normalized_lead = 1 - (avg_lead_time / self.price_increase_window) if avg_lead_time > 0 else 1

        # Pesi come nel backtest
        capture_weight = 0.4
        lead_weight = 0.25
        strength_weight = 0.15
        efficiency_weight = 0.2

        score = (
            efficiency * capture_weight +
            normalized_lead * lead_weight +
            min(avg_spike_strength / 10, 1) * strength_weight +
            efficiency * efficiency_weight
        )

        final_score = round(score, 4)
        logger.debug(f"🎯 Composite score: {final_score} (efficiency={efficiency:.3f}, lead={avg_lead_time:.2f}, strength={avg_spike_strength:.2f})")
        return final_score
    def get_empty_metrics(self) -> Dict[str, Any]:
        """Restituisce metriche vuote quando non ci sono dati"""
        return {
            'summary': {
                'total_spikes': 0,
                'ok_spikes': 0,
                'ko_spikes': 0,
                'efficiency': 0,
                'avg_price_increase': 0,
                'avg_lead_time': 0,
                'avg_spike_strength': 0,
                'percent_price_increasing': 0,
                'robustness_score': 0,
                'composite_score': 0,
                'final_score': 0
            },
            'summary_short': {
                'total_spikes': 0,
                'ok_spikes': 0,
                'ko_spikes': 0,
                'efficiency': 0,
                'avg_price_decrease': 0,
                'avg_lead_time_down': 0,
                'avg_spike_strength': 0,
                'robustness_score': 0,
                'composite_score': 0,
                'final_score': 0
            },
            'price_ranges': {
                '0-10%': 0, '10-20%': 0, '20-30%': 0,
                '30-50%': 0, '50-100%': 0, '>100%': 0
            },
            'price_ranges_short': {
                '0-10%': 0, '10-20%': 0, '20-30%': 0,
                '30-50%': 0, '50-100%': 0, '>100%': 0
            },
            'detailed_metrics': {
                'std_lead_time': 0,
                'min_lead_time': 0,
                'max_lead_time': 0,
                'min_spike_strength': 0,
                'max_spike_strength': 0
            },
            'detailed_metrics_short': {
                'std_lead_time_down': 0,
                'min_lead_time_down': 0,
                'max_lead_time_down': 0
            },
            'temporal_stats_long': {
                'by_hour': {h: 0 for h in range(24)},
                'by_weekday': {},
                'by_month': {m: 0 for m in range(1, 13)},
                'best_hour': None,
                'best_weekday_index': None,
                'best_weekday_name': None,
                'best_month': None
            },
            'temporal_stats_short': {
                'by_hour': {h: 0 for h in range(24)},
                'by_weekday': {},
                'by_month': {m: 0 for m in range(1, 13)},
                'best_hour': None,
                'best_weekday_index': None,
                'best_weekday_name': None,
                'best_month': None
            }
        }

    # 🔹 NUOVO: helper per costruire il DataFrame Excel identico per simulator e optimizer
    def build_spikes_excel_dataframe(self, all_spikes: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Costruisce il DataFrame con TUTTE le colonne e formattazioni
        usate per spikes_excel_*.csv (formato europeo, separatore ';').
        """
        csv_data = []
        for spike in all_spikes:
            csv_data.append({
                'symbol': spike['symbol'],
                'timestamp': str(int(spike['timestamp'])),  # ✅ Converti a intero e stringa
                'datetime': spike['datetime'],
                'volume_ratio': f"{spike['volume_ratio']:.6f}".replace('.', ','),  # ✅ Formato EU
                'current_volume': f"{spike['current_volume']:.2f}".replace('.', ','),  # ✅ 2 decimali
                'average_volume': f"{spike['average_volume']:.2f}".replace('.', ','),
                'current_price': f"{spike['current_price']:.8f}".replace('.', ','),
                'average_price': f"{spike['average_price']:.8f}".replace('.', ','),
                'price_ratio_at_spike': f"{spike['price_ratio']:.6f}".replace('.', ','),
                'price_increase': f"{spike['price_increase']:.6f}".replace('.', ','),
                'lead_time': str(spike['lead_time']),  # ✅ Converti a stringa
                'peak_price': f"{spike['peak_price']:.8f}".replace('.', ','),
                # Nuove colonne per Stop Loss / drawdown
                'price_decrease': f"{spike.get('price_decrease', 0.0):.6f}".replace('.', ','),
                'lead_time_down': str(spike.get('lead_time_down', 0)),
                'down_price': f"{spike.get('down_price', 0.0):.8f}".replace('.', ','),
                'is_ok': str(spike.get('is_ok', False)).upper(),  # ✅ TRUE/FALSE
                'is_ok_short': str(spike.get('is_ok_short', False)).upper(),  # ✅ TRUE/FALSE SHORT
                'direction': spike.get('direction', 'NONE'),
                'lookback_period': str(spike['lookback_period']),
                'price_increase_window': str(spike['price_increase_window']),
                'min_price_increase_required': f"{spike['min_price_increase_required']:.1f}".replace('.', ',')
            })

        df = pd.DataFrame(csv_data)
        return df

    def save_detailed_csv(self, all_spikes: List[Dict[str, Any]], metrics: Dict[str, Any]):
        """Salva CSV dettagliato con FORMATO NUMERICO CORRETTO"""
        logger.info("💾 Salvataggio risultati con formato numerico corretto...")

        os.makedirs('simulation_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        save_detailed = self.config.get('save_detailed_csv', True)
        save_us_csv = self.config.get('save_us_csv', True)

        # ✅ Usa helper condiviso per avere lo stesso formato ovunque
        df = self.build_spikes_excel_dataframe(all_spikes)

        # ✅ SALVA SEMPRE CSV FORMATO EXCEL (delimitatore ;, decimali ,) - MANDATORIO
        excel_filename = f'simulation_results/spikes_excel_{timestamp}.csv'
        df.to_csv(excel_filename, index=False, sep=';', quoting=csv.QUOTE_ALL)
        logger.info(f"✅ CSV Excel formato salvato: {excel_filename}")

        # ✅ SALVA CSV FORMATO US (delimitatore ,, decimali .) SOLO SE save_detailed_csv=1 E save_us_csv=1
        if save_detailed and save_us_csv:
            us_filename = f'simulation_results/spikes_us_{timestamp}.csv'
            df_us = df.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) and ',' in x else x)
            df_us.to_csv(us_filename, index=False, sep=',')
            logger.info(f"✅ CSV US formato salvato: {us_filename}")
        else:
            logger.info("⏭️  CSV US non salvato (SAVE_DETAILED_CSV=0 oppure SAVE_US_CSV=0)")

        # Salva metriche SOLO se save_detailed_csv=1
        if save_detailed:
            metrics_filename = f'simulation_results/metrics_{timestamp}.json'
            with open(metrics_filename, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"✅ Metrics JSON salvato: {metrics_filename}")
        else:
            logger.info("⏭️  Metrics JSON non salvato (SAVE_DETAILED_CSV=0)")

    def print_results(self, metrics: Dict[str, Any]):
        """Stampa risultati in formato leggibile"""
        summary = metrics['summary']
        price_ranges = metrics['price_ranges']

        summary_short = metrics.get('summary_short', {})
        price_ranges_short = metrics.get('price_ranges_short', {})
        temporal_long = metrics.get('temporal_stats_long', {})
        temporal_short = metrics.get('temporal_stats_short', {})

        print("\n" + "="*80)
        print("📊 RISULTATI SIMULAZIONE - SISTEMA LIVE REPLICA")
        print("="*80)
        print(f"📍 Parametri utilizzati:")
        print(f"   • Data Base Dir: {self.data_base_dir}")
        print(f"   • Volume Lookback: {self.volume_lookback}")
        print(f"   • Volume Threshold: {self.volume_threshold}")
        print(f"   • Volume Threshold MAX: {self.volume_threshold_max}")
        print(f"   • Finestra Prezzo: {self.price_increase_window} candele")
        print(f"   • Soglia Minima Aumento/Decremento: {self.min_price_increase_percent}%")
        print(f"   • Workers: {self.max_workers}, Batch: {self.batch_size}")
        print()
        print(f"📈 Metriche Performance LONG:")
        print(f"   • Spike Totali Rilevati: {summary['total_spikes']}")
        print(f"   • Eventi OK LONG: {summary['ok_spikes']}")
        print(f"   • Eventi NON-OK LONG: {summary['ko_spikes']}")
        print(f"   • Efficienza LONG: {summary['efficiency']*100:.2f}%")
        print(f"   • Aumento Prezzo Medio (OK LONG): {summary['avg_price_increase']:.2f}%")
        print(f"   • Lead Time Medio LONG: {summary['avg_lead_time']:.2f} candele")
        print(f"   • Prezzo in Aumento allo Spike: {summary['percent_price_increasing']*100:.2f}%")
        print()
        print(f"🎯 Score Avanzati LONG:")
        print(f"   • Affidabilità LONG: {summary['robustness_score']:.3f}/1.000")
        print(f"   • Score Complessivo LONG: {summary['final_score']:.3f}/1.000")
        print()
        print(f"📊 Distribuzione Aumenti Prezzo (solo OK LONG):")
        for range_name, count in price_ranges.items():
            percentage = (count / summary['ok_spikes'] * 100) if summary['ok_spikes'] > 0 else 0
            print(f"   • {range_name}: {count} eventi ({percentage:.1f}%)")

        if summary_short:
            print()
            print(f"📉 Metriche Performance SHORT:")
            print(f"   • Spike Totali Rilevati: {summary_short['total_spikes']}")
            print(f"   • Eventi OK SHORT: {summary_short['ok_spikes']}")
            print(f"   • Eventi NON-OK SHORT: {summary_short['ko_spikes']}")
            print(f"   • Efficienza SHORT: {summary_short['efficiency']*100:.2f}%")
            print(f"   • Decremento Prezzo Medio (OK SHORT): {summary_short['avg_price_decrease']:.2f}%")
            print(f"   • Lead Time Medio SHORT: {summary_short['avg_lead_time_down']:.2f} candele")
            print()
            print(f"🎯 Score Avanzati SHORT:")
            print(f"   • Affidabilità SHORT: {summary_short['robustness_score']:.3f}/1.000")
            print(f"   • Score Complessivo SHORT: {summary_short['final_score']:.3f}/1.000")

            if price_ranges_short:
                print()
                print(f"📊 Distribuzione Decrementi Prezzo (solo OK SHORT):")
                for range_name, count in price_ranges_short.items():
                    percentage = (count / summary_short['ok_spikes'] * 100) if summary_short['ok_spikes'] > 0 else 0
                    print(f"   • {range_name}: {count} eventi ({percentage:.1f}%)")

        print()
        print("🕒 Statistiche temporali LONG:")
        best_hour_long = temporal_long.get('best_hour')
        best_weekday_long = temporal_long.get('best_weekday_name')
        best_month_long = temporal_long.get('best_month')
        if best_hour_long is not None:
            print(f"   • Ora con più LONG: {best_hour_long:02d}:00")
        if best_weekday_long is not None:
            print(f"   • Giorno della settimana con più LONG: {best_weekday_long}")
        if best_month_long is not None:
            print(f"   • Mese con più LONG: {best_month_long}")

        print()
        print("🕒 Statistiche temporali SHORT:")
        best_hour_short = temporal_short.get('best_hour')
        best_weekday_short = temporal_short.get('best_weekday_name')
        best_month_short = temporal_short.get('best_month')
        if best_hour_short is not None:
            print(f"   • Ora con più SHORT: {best_hour_short:02d}:00")
        if best_weekday_short is not None:
            print(f"   • Giorno della settimana con più SHORT: {best_weekday_short}")
        if best_month_short is not None:
            print(f"   • Mese con più SHORT: {best_month_short}")

        print("="*80)

    # ================================
    #  NUOVE API CORE/SOFT PER OPTIMIZER
    # ================================
    def run_simulation_core_spikes(self) -> Dict[str, Any]:
        """
        Esegue solo la parte di rilevazione spike su TUTTI i simboli,
        restituendo ARRAYS compatti (Livello B) di spike DOPO TUTTI i filtri tranne
        quelli price_ratio_at_spike (MODE/MIN/MAX).

        Non salva CSV/JSON e non stampa i risultati aggregati.
        Usato dall'optimizer per riutilizzare gli spike su più combinazioni
        di filtri "soft" (PRICE_RATIO_AT_SPIKE_*).
        """
        logger.info("🚀 === AVVIO SCANSIONE CORE SPIKES (senza filtri price_ratio_at_spike) ===")

        # VERIFICA TIMEFRAME PRIMA DI INIZIARE
        if not self.check_timeframe_data_exists():
            raise FileNotFoundError(f"Dati per timeframe {self.timeframe} non trovati")

        symbols = self.get_available_symbols()

        self.start_time = time.time()
        self.symbols_processed = 0
        total_symbols = len(symbols)
        self.total_symbols = total_symbols

        logger.info(f"🚀 CORE: avvio analisi di {total_symbols} simboli in modalità sequenziale (riuso cache RAM/memmap)")

        acc_symbol = []
        acc_ts = []
        acc_dt_str = []
        acc_curr_vol = []
        acc_vol_ratio = []
        acc_price_ratio = []
        acc_hl_pct = []
        acc_body_ratio = []
        acc_is_ok = []
        acc_is_ok_short = []
        acc_price_increase = []
        acc_lead_time = []
        acc_price_decrease = []
        acc_lead_time_down = []

        total_spikes = 0
        core_progress_log_every = max(1, int(self.config.get("core_progress_log_every", 50)))
        core_subprofile = bool(self.config.get("core_subprofile", False))
        core_prof = {
            "load_np_s": 0.0,
            "extrema_s": 0.0,
            "candidate_build_s": 0.0,
            "candidate_loop_s": 0.0,
            "post_filter_s": 0.0,
            "price_prep_s": 0.0,
            "price_eval_s": 0.0,
            "datetime_s": 0.0,
        }

        for idx, symbol in enumerate(symbols, start=1):
            try:
                d = self.analyze_single_symbol_core_arrays(symbol)

                if core_subprofile:
                    dprof = d.get("__profile", {}) if isinstance(d, dict) else {}
                    for k in core_prof.keys():
                        core_prof[k] += float(dprof.get(k, 0.0))

                n_sym = int(len(d.get("ts", [])))
                if n_sym > 0:
                    acc_symbol.append(d["symbol"])
                    acc_ts.append(d["ts"])
                    acc_dt_str.append(d["dt_str"])
                    acc_curr_vol.append(d["curr_vol"])
                    acc_vol_ratio.append(d["vol_ratio"])
                    acc_price_ratio.append(d["price_ratio"])
                    acc_hl_pct.append(d["hl_pct"])
                    acc_body_ratio.append(d["body_ratio"])
                    acc_is_ok.append(d["is_ok"])
                    acc_is_ok_short.append(d["is_ok_short"])
                    acc_price_increase.append(d["price_increase"])
                    acc_lead_time.append(d["lead_time"])
                    acc_price_decrease.append(d["price_decrease"])
                    acc_lead_time_down.append(d["lead_time_down"])
                    total_spikes += n_sym

                self.symbols_processed = idx
                should_log_progress = (
                    idx == 1
                    or idx == total_symbols
                    or (idx % core_progress_log_every) == 0
                )
                if should_log_progress:
                    elapsed = time.time() - self.start_time
                    progress_pct = (self.symbols_processed / total_symbols) * 100 if total_symbols else 0.0
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                    logger.info(
                        f"✅ [CORE] Simbolo {idx}/{total_symbols} elaborato "
                        f"({progress_pct:.1f}% completato) - Spike cumulati: {total_spikes} - "
                        f"Memoria: {memory_usage:.1f}MB - Tempo trascorso: {elapsed:.1f}s"
                    )
            except Exception as e:
                logger.error(f"❌ [CORE] Errore analisi simbolo {symbol}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                continue

        def _cat_or_empty(parts, dtype):
            if not parts:
                return np.array([], dtype=dtype)
            return np.concatenate(parts)

        out = {
            "symbol": _cat_or_empty(acc_symbol, object),
            "ts": _cat_or_empty(acc_ts, np.int64),
            "dt_str": _cat_or_empty(acc_dt_str, object),
            "curr_vol": _cat_or_empty(acc_curr_vol, np.float64),
            "vol_ratio": _cat_or_empty(acc_vol_ratio, np.float64),
            "price_ratio": _cat_or_empty(acc_price_ratio, np.float64),
            "hl_pct": _cat_or_empty(acc_hl_pct, np.float64),
            "body_ratio": _cat_or_empty(acc_body_ratio, np.float64),
            "is_ok": _cat_or_empty(acc_is_ok, bool),
            "is_ok_short": _cat_or_empty(acc_is_ok_short, bool),
            "price_increase": _cat_or_empty(acc_price_increase, np.float64),
            "lead_time": _cat_or_empty(acc_lead_time, np.int32),
            "price_decrease": _cat_or_empty(acc_price_decrease, np.float64),
            "lead_time_down": _cat_or_empty(acc_lead_time_down, np.int32),
        }

        if core_subprofile:
            out["__core_profile"] = core_prof

        total_time = time.time() - self.start_time if self.start_time else 0.0
        if total_time > 0 and total_symbols:
            logger.info(
                f"🎯 Scansione CORE completata: {int(len(out['ts']))} spike in {total_time:.1f}s "
                f"({total_symbols / total_time:.2f} simboli/secondo)"
            )
        else:
            logger.info("🎯 Scansione CORE completata: nessun simbolo processato.")

        if core_subprofile:
            logger.info(
                "🧪 CORE SUBPROFILE: "
                f"load_np_s={core_prof['load_np_s']:.3f}, "
                f"extrema_s={core_prof['extrema_s']:.3f}, "
                f"candidate_build_s={core_prof['candidate_build_s']:.3f}, "
                f"candidate_loop_s={core_prof['candidate_loop_s']:.3f}, "
                f"post_filter_s={core_prof['post_filter_s']:.3f}, "
                f"price_prep_s={core_prof['price_prep_s']:.3f}, "
                f"price_eval_s={core_prof['price_eval_s']:.3f}, "
                f"datetime_s={core_prof['datetime_s']:.3f}"
            )

        return out

    def evaluate_spikes_with_current_filters(self, all_spikes_core: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Applica i filtri PRICE_RATIO_AT_SPIKE_* correnti agli spike core
        e calcola le metriche complete.

        NON salva CSV/JSON e NON stampa i risultati (usato dall'optimizer).

        Ritorna:
          - metrics: dict delle metriche
          - spikes_filtered: lista di spike dopo tutti i filtri (come in run_simulation)
        """
        if all_spikes_core is None:
            all_spikes_core = []

        # Copia shallow per sicurezza
        spikes = list(all_spikes_core)

        # Applica filtro price_ratio_at_spike secondo config corrente
        spikes_filtered = self._filter_spikes_by_price_ratio(spikes)

        # Calcola metriche esattamente come nel run_simulation standard
        metrics = self.calculate_metrics(spikes_filtered)

        return metrics, spikes_filtered

    def run_simulation(self) -> Dict[str, Any]:
        """Esegue la simulazione completa"""
        logger.info("🚀 === AVVIO SIMULAZIONE TRADING ULTRA-VELOCE ===")

        # VERIFICA TIMEFRAME PRIMA DI INIZIARE
        if not self.check_timeframe_data_exists():
            raise FileNotFoundError(f"Dati per timeframe {self.timeframe} non trovati")

        logger.info(f"⚙️  Parametri: Lookback={self.volume_lookback}, Threshold={self.volume_threshold}")
        logger.info(f"⚙️  Volume Threshold MAX: {self.volume_threshold_max}")
        logger.info(f"⚙️  Finestra prezzo: {self.price_increase_window} candele")
        logger.info(f"⚙️  Soglia minima aumento/decremento: {self.min_price_increase_percent}%")
        logger.info(f"⚙️  Filtro volume minimo: {self.min_current_volume}")
        logger.info(f"⚙️  Filtro volume massimo: {self.max_current_volume}")
        logger.info(f"⚙️  Filtro UTC Hour MIN (incl): {self.utc_hour_min}")
        logger.info(f"⚙️  Filtro UTC Hour MAX (escl): {self.utc_hour_max}")
        logger.info(f"⚙️  Filtro UTC_WEEKDAY (allowed): {sorted(list(self.utc_weekday_set)) if self.utc_weekday_set is not None else None}")
        logger.info(f"⚙️  Filtro price_ratio_at_spike MODE: {self.price_ratio_at_spike_mode}")
        logger.info(f"⚙️  Filtro price_ratio_at_spike MIN: {self.price_ratio_at_spike_min}")
        logger.info(f"⚙️  Filtro price_ratio_at_spike MAX: {self.price_ratio_at_spike_max}")
        logger.info(f"⚙️  HL_MIN_PCT: {self.hl_min_pct}")
        logger.info(f"⚙️  BODY_RATIO_MIN: {self.body_ratio_min}")
        logger.info(f"⚙️  Spike cooldown (candele): {self.spike_cooldown_candles}")
        logger.info(f"⚙️  Force sequential: {self.force_sequential}")

        start_time = time.time()

        # Analisi parallela di tutti i simboli
        all_spikes = self.analyze_all_symbols_parallel()

        # Calcolo metriche finali
        metrics = self.calculate_metrics(all_spikes)

        # Salva risultati
        self.save_detailed_csv(all_spikes, metrics)

        # Aggiungi informazioni temporali
        metrics['simulation_info'] = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'total_symbols_analyzed': self.total_symbols,
            'parameters_used': {
                'data_base_dir': self.data_base_dir,
                'data_memmap_dir': self.data_memmap_dir,
                'use_memmap': self.use_memmap,
                'volume_lookback': self.volume_lookback,
                'volume_threshold': self.volume_threshold,
                'volume_threshold_max': self.volume_threshold_max,
                'price_increase_window': self.price_increase_window,
                'min_price_increase_percent': self.min_price_increase_percent,
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
                'min_current_volume': self.min_current_volume,
                'max_current_volume': self.max_current_volume,
                'utc_hour_min': self.utc_hour_min,
                'utc_hour_max': self.utc_hour_max,
                'utc_weekday': sorted(list(self.utc_weekday_set)) if self.utc_weekday_set is not None else None,
                'price_ratio_at_spike_mode': self.price_ratio_at_spike_mode,
                'price_ratio_at_spike_min': self.price_ratio_at_spike_min,
                'price_ratio_at_spike_max': self.price_ratio_at_spike_max,
                'hl_min_pct': self.hl_min_pct,
                'body_ratio_min': self.body_ratio_min,
                'spike_cooldown_candles': self.spike_cooldown_candles,
                'force_sequential': self.force_sequential
            }
        }

        # Stampa risultati
        self.print_results(metrics)

        logger.info("🎉 Simulazione completata con successo!")
        return metrics

def load_simulator_config():
    """Carica configurazione da .env e simulator.env (simulator.env ha priorità)."""
    load_dotenv('.env', override=False)
    load_dotenv('simulator.env', override=True)

    volume_threshold_max_raw = os.getenv('VOLUME_THRESHOLD_MAX', '').strip()
    max_current_volume_raw = os.getenv('MAX_CURRENT_VOLUME', '').strip()

    utc_hour_min_raw = os.getenv('UTC_HOUR_MIN', '').strip()
    utc_hour_max_raw = os.getenv('UTC_HOUR_MAX', '').strip()

    utc_weekday_raw = os.getenv('UTC_WEEKDAY', '').strip()

    utc_weekday_list = None
    if utc_weekday_raw not in ("", None):
        # Parsing lista: "0,1,2,5" -> [0,1,2,5]
        parts = [p.strip() for p in utc_weekday_raw.split(',') if p.strip() != ""]
        if not parts:
            raise ValueError(
                "Configurazione non valida: UTC_WEEKDAY è settato ma vuoto dopo parsing. "
                "Esempio valido: UTC_WEEKDAY=0,1,2,5"
            )
        try:
            parsed = [int(p) for p in parts]
        except Exception:
            raise ValueError(
                f"Configurazione non valida: UTC_WEEKDAY deve contenere solo interi separati da virgola. Valore: {utc_weekday_raw}"
            )

        # Validazione range 0..6
        for d in parsed:
            if d < 0 or d > 6:
                raise ValueError(
                    f"Configurazione non valida: UTC_WEEKDAY contiene valore fuori range [0..6]: {d}. "
                    "Correggi simulator.env e riprova."
                )

        # Normalizza duplicati
        utc_weekday_list = sorted(list(set(parsed)))

    # Filtri opzionali su price_ratio_at_spike
    price_ratio_min_raw = os.getenv('PRICE_RATIO_AT_SPIKE_MIN', '').strip()
    price_ratio_max_raw = os.getenv('PRICE_RATIO_AT_SPIKE_MAX', '').strip()

    price_ratio_at_spike_min = (float(price_ratio_min_raw) if price_ratio_min_raw not in ("", None) else None)
    price_ratio_at_spike_max = (float(price_ratio_max_raw) if price_ratio_max_raw not in ("", None) else None)

    # MODALITÀ FILTRO price_ratio
    price_ratio_mode_raw = os.getenv('PRICE_RATIO_AT_SPIKE_MODE', '').strip()
    price_ratio_mode = (price_ratio_mode_raw if price_ratio_mode_raw not in ("", None) else 'TAILS')
    price_ratio_mode = str(price_ratio_mode).strip().upper()

    if price_ratio_mode not in ('TAILS', 'RANGE'):
        raise ValueError(
            f"Configurazione non valida: PRICE_RATIO_AT_SPIKE_MODE deve essere TAILS o RANGE. "
            f"Valore: {price_ratio_mode_raw if price_ratio_mode_raw not in ('', None) else 'TAILS (default)'}"
        )

    # Validazione a monte per MODE=RANGE (coerente con __init__)
    if price_ratio_mode == 'RANGE':
        if price_ratio_at_spike_min is None or price_ratio_at_spike_max is None:
            raise ValueError(
                "Configurazione non valida: con PRICE_RATIO_AT_SPIKE_MODE=RANGE devi settare "
                "sia PRICE_RATIO_AT_SPIKE_MIN sia PRICE_RATIO_AT_SPIKE_MAX. Correggi simulator.env e riprova."
            )
        if float(price_ratio_at_spike_min) >= float(price_ratio_at_spike_max):
            raise ValueError(
                f"Configurazione non valida: con PRICE_RATIO_AT_SPIKE_MODE=RANGE devi avere "
                f"PRICE_RATIO_AT_SPIKE_MIN ({price_ratio_at_spike_min}) < PRICE_RATIO_AT_SPIKE_MAX ({price_ratio_at_spike_max}). "
                "Correggi simulator.env e riprova."
            )

    hl_min_pct_raw = os.getenv('HL_MIN_PCT', '').strip()
    body_ratio_min_raw = os.getenv('BODY_RATIO_MIN', '').strip()

    hl_min_pct = (float(hl_min_pct_raw) if hl_min_pct_raw not in ("", None) else None)
    body_ratio_min = (float(body_ratio_min_raw) if body_ratio_min_raw not in ("", None) else None)

    use_memmap_raw = os.getenv('USE_MEMMAP', '').strip()
    use_memmap = True
    if use_memmap_raw not in ("", None):
        use_memmap = (use_memmap_raw.strip() == '1')

    config = {
        # D6=B FIX: base dir dati storico configurabile (per esecuzione in cwd trial)
        'data_base_dir': os.getenv('DATA_BASE_DIR', 'data/historical'),

        # MEMMAP: base dir per i .npy (1 file per simbolo/timeframe in directory)
        'data_memmap_dir': os.getenv('DATA_MEMMAP_DIR', 'data/memmap'),
        'use_memmap': use_memmap,

        'volume_lookback': int(os.getenv('VOLUME_LOOKBACK', '5')),
        'volume_threshold': float(os.getenv('VOLUME_THRESHOLD', '3.0')),
        'volume_threshold_max': (float(volume_threshold_max_raw) if volume_threshold_max_raw not in ("", None) else None),
        'price_increase_window': int(os.getenv('PRICE_INCREASE_WINDOW', '5')),
        'min_price_increase_percent': float(os.getenv('MIN_PRICE_INCREASE_PERCENT', '2.0')),
        'min_current_volume': float(os.getenv('MIN_CURRENT_VOLUME', '0')),  # NUOVO PARAMETRO
        'max_current_volume': (float(max_current_volume_raw) if max_current_volume_raw not in ("", None) else None),  # NUOVO PARAMETRO
        'utc_hour_min': (int(utc_hour_min_raw) if utc_hour_min_raw not in ("", None) else None),  # NUOVO PARAMETRO
        'utc_hour_max': (int(utc_hour_max_raw) if utc_hour_max_raw not in ("", None) else None),  # NUOVO PARAMETRO
        'utc_weekday': utc_weekday_list,  # NUOVO PARAMETRO (lista di int o None)
        'timeframe': os.getenv('TIMEFRAME', '1h'),
        'max_workers': int(os.getenv('MAX_WORKERS', str(min(mp.cpu_count(), 12)))),
        'batch_size': int(os.getenv('BATCH_SIZE', '25')),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_to_file': os.getenv('LOG_TO_FILE', '1').strip() == '1',
        'save_us_csv': os.getenv('SAVE_US_CSV', '1').strip() == '1',
        'save_detailed_csv': os.getenv('SAVE_DETAILED_CSV', '1').strip() == '1',
        # NUOVO: cooldown configurabile
        'spike_cooldown_candles': int(os.getenv('SPIKE_COOLDOWN_CANDLES', '0')),

        # Filtri opzionali su price_ratio_at_spike + MODE
        'price_ratio_at_spike_mode': price_ratio_mode,
        'price_ratio_at_spike_min': price_ratio_at_spike_min,
        'price_ratio_at_spike_max': price_ratio_at_spike_max,

        # NUOVI: filtri su candela spike
        'hl_min_pct': hl_min_pct,
        'body_ratio_min': body_ratio_min,

        # NUOVO: forza esecuzione sequenziale (per uso con optimizer multiprocess)
        'force_sequential': os.getenv('SIMULATOR_FORCE_SEQUENTIAL', '0').strip() == '1',
        'core_extrema_mode': os.getenv('CORE_EXTREMA_MODE', 'candidate').strip().lower(),
        'core_use_numba': os.getenv('CORE_USE_NUMBA', '1').strip() == '1',
    }

    # RICONFIGURA IL LOGGING CON IL LIVELLO CORRETTO + FILE OPZIONALE
    global logger
    logger = setup_detailed_logging(config['log_level'], log_to_file=config.get('log_to_file', True))

    logger.info("📋 Configurazione caricata:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")

    # Diagnostica esplicita Numba all'avvio (richiesta utente).
    numba_version = "not-installed"
    if NUMBA_AVAILABLE:
        try:
            import numba as _numba
            numba_version = getattr(_numba, "__version__", "unknown")
        except Exception:
            numba_version = "available-but-version-unreadable"

    numba_requested = config.get('core_use_numba', False)
    numba_active = bool(numba_requested and NUMBA_AVAILABLE)
    if numba_active:
        numba_reason = "JIT attivo"
    elif numba_requested and not NUMBA_AVAILABLE:
        numba_reason = "CORE_USE_NUMBA=1 ma numba non è installato"
    else:
        numba_reason = "CORE_USE_NUMBA=0"

    logger.info(
        "⚙️ CORE NUMBA STATUS | requested=%s | available=%s | active=%s | version=%s | reason=%s",
        numba_requested,
        NUMBA_AVAILABLE,
        numba_active,
        numba_version,
        numba_reason,
    )

    return config

def main():
    """Funzione principale"""
    try:
        # Carica configurazione e setup logging
        config = load_simulator_config()

        # Avvia simulatore
        simulator = TradingSimulator(config)
        results = simulator.run_simulation()

        logger.info("🎊 Simulazione completata con successo!")
        return results

    except Exception as e:
        logger.error(f"💥 Errore durante la simulazione: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    # Fix per Windows
    if mp.get_start_method() == 'spawn':
        mp.set_start_method('spawn', force=True)

    main()
