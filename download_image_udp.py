try:
    import eventlet  # type: ignore

    eventlet.monkey_patch()
    _ASYNC_MODE = "eventlet"
except Exception:
    eventlet = None
    _ASYNC_MODE = "threading"

import collections
import socket
import pickle
import struct
import numpy as np
from image_detach_rebuild import redraw_image, PIECE_SIZE, PATCH_PIECE_SIZE as NEURAL_BYPASS_PIECE_SIZE
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from PIL import Image
import base64
import io
import os
import threading
import time
import random
from typing import Optional

# Configuration
HOST = 'localhost'
PORT = 10010
GNURADIO_PDU_PORT = 50010  # GNU Radio IRS_tranceiver network_socket_pdu input
# socket_pdu MTU in gnu_radio/*.grc (UDP receive buffer); not the WiFi MAC limit.
GNURADIO_SOCKET_MTU = 65507
# ieee802_11 MAC rejects application payloads larger than this (see mac block error).
GNURADIO_MAC_MSDU_MAX = 1500
IMAGE_SIZE = (300, 300, 3)
UPLOAD_FOLDER = 'uploads'
# JSCE (same defaults as upload_featuremap_udp.py / download_featuremap_udp.py)
CODEC_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "codec",
    "checkpoints",
    "Rician-checkpoint_SOMA-DSCN-exp-ver_noIRS_fixIRS_AP-1_Usr-5_img-size-128_epoch100_20250306.pth",
)
CODEC_IMG_SIZE = (240, 240)
CODEC_COMPRESSED_CH = 128
LATENT_SHAPE = (
    CODEC_IMG_SIZE[0] // 8,
    CODEC_IMG_SIZE[1] // 8,
    CODEC_COMPRESSED_CH,
)
# Latent patch auto-sized to fill ~1500 B 802.11 MSDU (see _resolve_gnuradio_latent_piece_size).
GNURADIO_LATENT_WIRE_DTYPE = np.float16
GNURADIO_LATENT_PREVIEW_MIN_INTERVAL_S = 0.4
_gnuradio_latent_piece_size_cache: Optional[tuple[int, int]] = None
_gnuradio_latent_preview_milestone: int = -1
_gnuradio_latent_last_preview_ts: float = 0.0
_jsce_codec = None
_jsce_lock = threading.Lock()

# HuggingFace dataset cache directory (within project dir)
HF_DATASET_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hf_cache",
)
# Time delay history for computation-time visualization
time_delay_data: dict[str, list[dict]] = {}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=_ASYNC_MODE)
stop_thread = False  # receiver stop flag
receiver_task_running = False
# multi-link reconstruction (keyed by CSI like "13-14")
reconstructed_images: dict[str, np.ndarray] = {}
# GNU Radio + JSCE: persistent merged latent; patches update regions in-place.
active_gnuradio_csis: list[str] = []
_gnuradio_rx_lock = threading.Lock()
_gnuradio_rx_csi_queue: collections.deque[str] = collections.deque()
merged_latent = np.zeros(LATENT_SHAPE, dtype=np.float32)
_gnuradio_latent_lock = threading.Lock()
# Latent round tracking (patches vs grid cells are different counts — do not mix them).
_gnuradio_latent_received_keys: set[tuple[int, int]] = set()
_gnuradio_latent_patches_received: int = 0
_gnuradio_latent_patches_expected: int = 0
_gnuradio_latent_cells_expected: int = LATENT_SHAPE[0] * LATENT_SHAPE[1]
_gnuradio_latent_tx_generation: int = 0
_gnuradio_latent_rx_generation: int = 0
_gnuradio_latent_round_decoded: bool = False

# --- quantitative metrics ---
original_images: dict[str, np.ndarray] = {}
metrics_state: dict[str, dict] = {}
_metrics_last_emit: dict[str, float] = {}

send_stop_flag = threading.Event()

# --- multi-link transmitter state (Proteus-like canvas) ---
# node: {id, x, y, label}
nodes_state = [
    {"id": "3", "x": 90, "y": 80, "label": "3"},
    {"id": "4", "x": 230, "y": 110, "label": "4"},
    {"id": "10", "x": 130, "y": 230, "label": "10"},
    {"id": "13", "x": 280, "y": 240, "label": "13"},
]

# default static CSI from your existing scripts (`upload_featuremap_udp.py`/`download_featuremap_udp.py`)
selected_csis: list[str] = ["3-4", "13-10"]
channel_states: dict[str, dict] = {}
# When False, latent/RGB pieces pass through without toy drop/noise (real GNU Radio path only).
_sim_channel_effect_enabled: bool = True
# Slightly slower Socket.IO emit after each UDP piece so the browser can paint incremental rebuilds.
RX_PIECE_EMIT_DELAY_S = 0.03
UDP_RECV_BUF = 65535


def _len_prefixed_packet(payload: bytes) -> bytes:
    return struct.pack("=L", len(payload)) + payload


def _pickle_jsce_latent_payload(
        piece,
        tx_gen: int = 0,
        expected_patches: int = 0,
) -> bytes:
    (pos, arr) = piece
    arr_wire = np.asarray(arr, dtype=GNURADIO_LATENT_WIRE_DTYPE)
    return pickle.dumps(
        {
            "payload_type": "jsce_latent",
            "piece": (pos, arr_wire),
            "wire_dtype": str(GNURADIO_LATENT_WIRE_DTYPE),
            "tx_gen": int(tx_gen),
            "expected_patches": int(expected_patches),
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _jsce_latent_meta_from_obj(obj: dict) -> dict:
    return {
        "tx_gen": int(obj.get("tx_gen", 0)),
        "expected_patches": int(obj.get("expected_patches", 0)),
    }


def _normalize_latent_piece_from_wire(piece):
    (pos, arr) = piece
    return (pos, np.asarray(arr, dtype=np.float32))


def _estimate_gnuradio_latent_piece_packet_bytes(piece_h: int, piece_w: int) -> int:
    """Wire size for one spatial JSCE latent patch (float16 on wire) + 4-byte length header."""
    piece = (((0, 0, 0), np.zeros((piece_h, piece_w, CODEC_COMPRESSED_CH), dtype=GNURADIO_LATENT_WIRE_DTYPE)))
    return len(_len_prefixed_packet(_pickle_jsce_latent_payload(piece)))


def _find_optimal_gnuradio_latent_piece_size(mtu: int = GNURADIO_MAC_MSDU_MAX) -> tuple[int, int]:
    """Largest (h, w) patch whose pickled float16 payload fits in the 802.11 MSDU budget."""
    lh, lw = LATENT_SHAPE[0], LATENT_SHAPE[1]
    best = (1, 1)
    best_bytes = 0
    for ph in range(1, lh + 1):
        for pw in range(1, lw + 1):
            nbytes = _estimate_gnuradio_latent_piece_packet_bytes(ph, pw)
            if nbytes > mtu:
                continue
            area = ph * pw
            if nbytes > best_bytes or (nbytes == best_bytes and area > best[0] * best[1]):
                best_bytes = nbytes
                best = (ph, pw)
    return best


def _resolve_gnuradio_latent_piece_size(
        user_hint: Optional[tuple[int, int]] = None,
        mtu: int = GNURADIO_MAC_MSDU_MAX,
) -> tuple[int, int]:
    """
    Pick the largest MTU-filling latent patch, optionally capped by user_hint dimensions.
    Result is cached until process restart.
    """
    global _gnuradio_latent_piece_size_cache
    optimal = _find_optimal_gnuradio_latent_piece_size(mtu)
    if user_hint is None:
        if _gnuradio_latent_piece_size_cache is None:
            _gnuradio_latent_piece_size_cache = optimal
        return _gnuradio_latent_piece_size_cache

    uh, uw = int(user_hint[0]), int(user_hint[1])
    chosen = (
        max(1, min(optimal[0], uh, LATENT_SHAPE[0])),
        max(1, min(optimal[1], uw, LATENT_SHAPE[1])),
    )
    while _estimate_gnuradio_latent_piece_packet_bytes(chosen[0], chosen[1]) > mtu:
        if chosen[0] > 1:
            chosen = (chosen[0] - 1, chosen[1])
        elif chosen[1] > 1:
            chosen = (chosen[0], chosen[1] - 1)
        else:
            break
    return chosen


def _max_gnuradio_latent_piece_side(mtu: int = GNURADIO_MAC_MSDU_MAX) -> int:
    ph, _pw = _find_optimal_gnuradio_latent_piece_size(mtu)
    return ph


def _coerce_gnuradio_latent_piece_size(piece_size: tuple[int, int]) -> tuple[int, int]:
    """Backward-compatible alias: cap optimal patch by UI hint."""
    return _resolve_gnuradio_latent_piece_size(user_hint=piece_size)


def _expected_latent_pieces(piece_size: tuple[int, int]) -> int:
    ph, pw = piece_size
    lh, lw = LATENT_SHAPE[0], LATENT_SHAPE[1]
    return max(1, (lh // ph) * (lw // pw))


def _gnuradio_latent_piece_size_error(piece_size: tuple[int, int], mtu: int = GNURADIO_MAC_MSDU_MAX) -> Optional[str]:
    ph, pw = piece_size
    packet_bytes = _estimate_gnuradio_latent_piece_packet_bytes(ph, pw)
    if packet_bytes <= mtu:
        return None
    opt = _find_optimal_gnuradio_latent_piece_size(mtu)
    return (
        f"JSCE latent patch {ph}x{pw} (all {CODEC_COMPRESSED_CH} ch, float16 wire) produces "
        f"~{packet_bytes} B packets, but the GNU Radio 802.11 MAC limit is {mtu} B per frame. "
        f"Max fitting patch is {opt[0]}x{opt[1]} (~{_estimate_gnuradio_latent_piece_packet_bytes(*opt)} B)."
    )


def _latent_spatial_coverage_keys(piece) -> set[tuple[int, int]]:
    """Grid cells (y, x) filled by one latent patch (multi- or single-channel)."""
    (y, x, c), arr = piece
    ph, pw = int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3 and int(arr.shape[2]) == 1:
        return {(int(y), int(x))}
    keys: set[tuple[int, int]] = set()
    for dy in range(ph):
        for dx in range(pw):
            keys.add((int(y) + dy, int(x) + dx))
    return keys


def _safe_unpack_len_prefixed(data: bytes) -> bytes:
    """
    Some senders prepend a 4-byte little-endian length header: struct.pack("=L", len(payload)) + payload.
    This helper strips that header when it matches the remaining payload size.
    """
    if len(data) < 4:
        return data
    try:
        (n,) = struct.unpack("=L", data[:4])
    except struct.error:
        return data
    if 0 <= n == len(data[4:]):
        return data[4:]
    return data


def _is_latent_piece_array(arr) -> bool:
    if not hasattr(arr, "shape") or len(arr.shape) != 3:
        return False
    if not np.issubdtype(getattr(arr, "dtype", np.float32), np.floating):
        return False
    ch = int(arr.shape[2])
    return ch == 1 or ch == CODEC_COMPRESSED_CH


def _latent_piece_message(piece, tx_gen: int = 0, expected_patches: int = 0) -> dict:
    return {
        "piece": _normalize_latent_piece_from_wire(piece),
        "tx_gen": int(tx_gen),
        "expected_patches": int(expected_patches),
    }


def _try_decode_udp(data: bytes):
    """
    Returns one of:
      ("piece_csi", (csi, piece))
      ("piece", piece)
      ("latent_piece", piece)
      ("jpeg", pil_image_rgb)
      ("none", None)
    """
    payload = _safe_unpack_len_prefixed(data)

    # 1) pickle piece / JSCE latent
    try:
        obj = pickle.loads(payload)

        if isinstance(obj, dict) and obj.get("payload_type") == "jsce_latent" and "piece" in obj:
            piece = obj.get("piece")
            if isinstance(piece, tuple) and len(piece) == 2 and isinstance(piece[0], tuple) and len(piece[0]) == 3:
                return (
                    "latent_piece",
                    _latent_piece_message(piece, **_jsce_latent_meta_from_obj(obj)),
                )

        # multi-link RGB formats (legacy / neural bypass over UDP):
        #  - {"csi": "13-14", "piece": ((y,x,c), arr)}
        #  - ("13-14", ((y,x,c), arr))
        if isinstance(obj, dict) and "piece" in obj and "csi" in obj:
            csi = str(obj.get("csi"))
            piece = obj.get("piece")
            if isinstance(piece, tuple) and len(piece) == 2 and isinstance(piece[0], tuple) and len(piece[0]) == 3:
                if _is_latent_piece_array(piece[1]):
                    return ("latent_piece", _latent_piece_message(piece))
                return ("piece_csi", (csi, piece))
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str):
            csi = obj[0]
            piece = obj[1]
            if isinstance(piece, tuple) and len(piece) == 2 and isinstance(piece[0], tuple) and len(piece[0]) == 3:
                if _is_latent_piece_array(piece[1]):
                    return ("latent_piece", _latent_piece_message(piece))
                return ("piece_csi", (csi, piece))

        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], tuple) and len(obj[0]) == 3:
            if _is_latent_piece_array(obj[1]):
                return ("latent_piece", _latent_piece_message(obj))
            return ("piece", obj)
    except Exception:
        pass

    # 2) raw JPEG/PNG bytes (e.g., from GNU Radio path)
    try:
        img = Image.open(io.BytesIO(payload)).convert("RGB")
        return ("jpeg", img)
    except Exception:
        return ("none", None)


def _compute_channel_state_for_csi(csi: str):
    """
    Toy CSI model:
    - SNR depends on node distance (path loss)
    - drop probability grows with distance
    - random seed derived from (src,dst) for repeatability
    """
    # csi format "13-14"
    try:
        src, dst = csi.split("-", 1)
    except ValueError:
        channel_states[csi] = {"snr_db": 25.0, "seed": 42, "drop_prob": 0.0, "drop_prob_extra": 0.0}
        return
    n1 = next((n for n in nodes_state if n["id"] == src), None)
    n2 = next((n for n in nodes_state if n["id"] == dst), None)
    if not n1 or not n2:
        channel_states[csi] = {"snr_db": 25.0, "seed": 42, "drop_prob": 0.0, "drop_prob_extra": 0.0}
        return

    dx = float(n1["x"] - n2["x"])
    dy = float(n1["y"] - n2["y"])
    dist = (dx * dx + dy * dy) ** 0.5

    # Canvas is ~ (0..420). Map distance to a rough SNR range.
    snr_db = max(3.0, min(35.0, 35.0 - dist / 18.0))
    drop_canvas = max(0.0, min(0.35, (dist - 80.0) / 600.0))
    seed = (hash(f"{src}->{dst}") ^ 0x5BD1E995) & 0xFFFFFFFF
    extra = float(channel_states.get(csi, {}).get("drop_prob_extra", 0.0))
    drop_eff = min(0.95, drop_canvas + extra)

    channel_states[csi] = {
        "snr_db": float(snr_db),
        "seed": int(seed),
        "drop_prob_canvas": float(drop_canvas),
        "drop_prob_extra": extra,
        "drop_prob": float(drop_eff),
    }


# --- Quantitative metrics helpers --------------------------------------------------

def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    diff = img1[:h, :w].astype(np.float64) - img2[:h, :w].astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = compute_mse(img1, img2)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    a = img1[:h, :w].astype(np.float64)
    b = img2[:h, :w].astype(np.float64)
    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2
    ssim_c = []
    for c in range(3):
        aa, bb = a[:, :, c], b[:, :, c]
        mu1, mu2 = np.mean(aa), np.mean(bb)
        s1, s2 = np.var(aa), np.var(bb)
        s12 = np.mean((aa - mu1) * (bb - mu2))
        num = (2.0 * mu1 * mu2 + C1) * (2.0 * s12 + C2)
        den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
        ssim_c.append(num / den)
    return float(np.mean(ssim_c))


def init_metrics(csi: str, expected_pieces: int = 0):
    metrics_state.setdefault(csi, {
        "piece_count": 0,
        "total_bytes": 0,
        "dropped_count": 0,
        "start_time": time.time(),
        "last_ssim": 0.0,
        "last_psnr": 0.0,
        "ssim_history": [],
        "psnr_history": [],
        "throughput_history": [],
        "time_history": [],
        "expected_pieces": expected_pieces,
        "_last_piece_time": 0.0,
    })


def _update_piece_metrics(csi: str, piece_bytes: int = 0, dropped: bool = False):
    st = metrics_state.get(csi)
    if st is None:
        init_metrics(csi)
        st = metrics_state[csi]
    st["piece_count"] += 1
    if not dropped:
        st["total_bytes"] += piece_bytes
    else:
        st["dropped_count"] += 1
    st["_last_piece_time"] = time.time()


def _update_quality_metrics(csi: str, reconstructed: np.ndarray, original: np.ndarray):
    st = metrics_state.get(csi)
    if st is None:
        return
    ssim = compute_ssim(reconstructed, original)
    psnr = compute_psnr(reconstructed, original)
    st["last_ssim"] = ssim
    st["last_psnr"] = psnr
    elapsed = time.time() - st["start_time"]
    st["ssim_history"].append(ssim)
    st["psnr_history"].append(psnr)
    st["time_history"].append(elapsed)


def _compute_throughput(csi: str) -> float:
    st = metrics_state.get(csi)
    if st is None or st["piece_count"] < 2:
        return 0.0
    elapsed = time.time() - st["start_time"]
    return st["total_bytes"] / elapsed if elapsed > 0 else 0.0


def _emit_metrics(csi: str):
    st = metrics_state.get(csi)
    if st is None:
        return
    now = time.time()
    # throttle to ~every 400 ms
    last = _metrics_last_emit.get(csi, 0.0)
    if now - last < 0.4 and st["piece_count"] % 5 != 0:
        return
    _metrics_last_emit[csi] = now
    elapsed = now - st["start_time"]
    throughput = _compute_throughput(csi)
    completion = 0.0
    if st["expected_pieces"] > 0:
        completion = min(100.0, 100.0 * st["piece_count"] / st["expected_pieces"])
    data = {
        "csi": csi,
        "piece_count": st["piece_count"],
        "dropped_count": st["dropped_count"],
        "elapsed_seconds": round(elapsed, 1),
        "throughput_bps": round(throughput, 1),
        "ssim": round(st["last_ssim"], 4),
        "psnr": round(st["last_psnr"], 2),
        "completion_pct": round(completion, 1),
        "expected_pieces": st["expected_pieces"],
        "time_history": st["time_history"][-100:],
        "ssim_history": [round(v, 4) for v in st["ssim_history"][-100:]],
        "psnr_history": [round(v, 2) for v in st["psnr_history"][-100:]],
    }
    socketio.emit("metrics_update", data)


def _estimate_piece_bytes(csi: str, piece) -> int:
    """Estimate wire bytes for a piece (payload, not UDP header)."""
    return int(piece[1].nbytes) if hasattr(piece[1], "nbytes") else int(np.prod(piece[1].shape))


# -----------------------------------------------------------------------------------

def _form_flag(name: str, default: bool = False) -> bool:
    raw = request.form.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "on", "true", "yes")


def _apply_channel_effect(piece, csi: str):
    """
    Apply a lightweight, visually-obvious "channel" effect to a piece:
    - random drop (piece not delivered — caller should skip update)
    - additive gaussian noise scaled by SNR

    Drop is decided independently on each transmission attempt so the same
    spatial patch is not permanently lost due to a fixed (y, x) seed.
    Returns None when the piece is dropped.
    """
    if not _sim_channel_effect_enabled:
        return piece

    (y, x, c), val = piece

    st = channel_states.get(csi) or {"snr_db": 25.0, "seed": 42, "drop_prob": 0.0}

    if random.random() < float(st.get("drop_prob", 0.0)):
        return None

    snr_db = float(st.get("snr_db", 25.0))
    # signal assumed in [0,255]; noise sigma shrinks with SNR
    sigma = max(1.0, 22.0 * (10.0 ** (-snr_db / 20.0)))
    noise = np.random.normal(0.0, sigma, size=val.shape).astype(np.float32)
    out = np.clip(val.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return ((y, x, c), out)


def _apply_channel_effect_latent(piece, csi: str):
    """Toy channel on JSCE latent patches (float32); returns None when dropped."""
    if not _sim_channel_effect_enabled:
        return piece

    (y, x, c), val = piece
    st = channel_states.get(csi) or {"snr_db": 25.0, "seed": 42, "drop_prob": 0.0}
    if random.random() < float(st.get("drop_prob", 0.0)):
        return None
    snr_db = float(st.get("snr_db", 25.0))
    sigma = max(0.005, 0.35 * (10.0 ** (-snr_db / 20.0)))
    noise = np.random.normal(0.0, sigma, size=val.shape).astype(np.float32)
    out = val.astype(np.float32) + noise
    return ((y, x, c), out)


def _begin_gnuradio_latent_round(tx_gen: int, expected_patches: int):
    """Reset per-round RX counters only; merged_latent is never cleared."""
    global _gnuradio_latent_received_keys, _gnuradio_latent_patches_received
    global _gnuradio_latent_patches_expected, _gnuradio_latent_rx_generation
    global _gnuradio_latent_round_decoded, _gnuradio_latent_preview_milestone
    global _gnuradio_latent_last_preview_ts
    with _gnuradio_latent_lock:
        _gnuradio_latent_received_keys.clear()
        _gnuradio_latent_patches_received = 0
        _gnuradio_latent_patches_expected = max(0, int(expected_patches))
        _gnuradio_latent_rx_generation = int(tx_gen)
        _gnuradio_latent_round_decoded = False
        _gnuradio_latent_preview_milestone = -1
        _gnuradio_latent_last_preview_ts = 0.0


def _reset_merged_latent_state(expected_pieces: int = 0, clear_keys: bool = True):
    """Reset per-round counters only; merged_latent buffer is preserved."""
    _begin_gnuradio_latent_round(
        tx_gen=0 if clear_keys else _gnuradio_latent_rx_generation,
        expected_patches=expected_pieces,
    )


def _apply_gnuradio_latent_piece(piece, tx_gen: int = 0, expected_patches: int = 0) -> Optional[dict]:
    """
    Merge one latent patch into the persistent buffer (in-place by spatial region).
    Returns progress stats for the current TX round.
    """
    global merged_latent, _gnuradio_latent_patches_received, _gnuradio_latent_patches_expected

    if tx_gen > 0 and tx_gen != _gnuradio_latent_rx_generation:
        _begin_gnuradio_latent_round(tx_gen, expected_patches)
    elif expected_patches > 0 and _gnuradio_latent_patches_expected == 0:
        _gnuradio_latent_patches_expected = int(expected_patches)

    with _gnuradio_latent_lock:
        merged_latent = redraw_image(piece, merged_latent)
        _gnuradio_latent_received_keys.update(_latent_spatial_coverage_keys(piece))
        _gnuradio_latent_patches_received += 1
        return {
            "n_patches": _gnuradio_latent_patches_received,
            "n_patches_exp": _gnuradio_latent_patches_expected,
            "n_cells": len(_gnuradio_latent_received_keys),
            "n_cells_exp": _gnuradio_latent_cells_expected,
        }


def _latent_coverage_complete() -> bool:
    with _gnuradio_latent_lock:
        exp = _gnuradio_latent_patches_expected
        if exp <= 0 or _gnuradio_latent_round_decoded:
            return False
        return _gnuradio_latent_patches_received >= exp


def _maybe_preview_decode_gnuradio_latent(csis: list[str], n_cells: int, n_cells_exp: int):
    """
    Periodic partial msg2img for UX (preview quality improves as grid coverage grows).
    Uses grid-cell coverage for progress; final decode uses patch count.
    """
    global _gnuradio_latent_preview_milestone, _gnuradio_latent_last_preview_ts
    if n_cells_exp <= 0 or _latent_coverage_complete():
        return
    min_cells = max(4, n_cells_exp // 25)
    if n_cells < min_cells:
        return
    milestone = min(9, int(10.0 * n_cells / n_cells_exp))
    if milestone <= _gnuradio_latent_preview_milestone:
        return
    now = time.time()
    if now - _gnuradio_latent_last_preview_ts < GNURADIO_LATENT_PREVIEW_MIN_INTERVAL_S:
        return
    _gnuradio_latent_preview_milestone = milestone
    _gnuradio_latent_last_preview_ts = now
    with _gnuradio_latent_lock:
        latent_snapshot = merged_latent.copy()
    pct = int(100.0 * n_cells / n_cells_exp)
    print(f"JSCE latent preview decode at ~{pct}% grid ({n_cells}/{n_cells_exp} cells)")
    _decode_and_emit_latent_for_csis(latent_snapshot, csis, is_piece=True)


def _maybe_decode_complete_gnuradio_latent(csis: list[str]) -> bool:
    """Run msg2img once all UDP patches for the round have arrived."""
    if not _latent_coverage_complete():
        return False
    global _gnuradio_latent_round_decoded
    with _gnuradio_latent_lock:
        n_patches = _gnuradio_latent_patches_received
        n_exp = _gnuradio_latent_patches_expected
        n_cells = len(_gnuradio_latent_received_keys)
        latent_snapshot = merged_latent.copy()
        _gnuradio_latent_round_decoded = True
    print(
        f"JSCE latent complete ({n_patches}/{n_exp} patches, "
        f"{n_cells}/{_gnuradio_latent_cells_expected} cells) — final decode"
    )
    _decode_and_emit_latent_for_csis(latent_snapshot, csis, is_piece=False)
    return True


def _reset_gnuradio_rx_state(csis: list[str]):
    """Track active GNU Radio links and reset JSCE latent RX state."""
    global active_gnuradio_csis
    active_gnuradio_csis = [str(c) for c in csis]
    with _gnuradio_rx_lock:
        _gnuradio_rx_csi_queue.clear()
    _reset_merged_latent_state()


def _queue_gnuradio_rx_csi(csi: str):
    with _gnuradio_rx_lock:
        _gnuradio_rx_csi_queue.append(str(csi))


def _resolve_gnuradio_rx_csi() -> str:
    with _gnuradio_rx_lock:
        if _gnuradio_rx_csi_queue:
            return _gnuradio_rx_csi_queue.popleft()
    if len(active_gnuradio_csis) == 1:
        return active_gnuradio_csis[0]
    if active_gnuradio_csis:
        return active_gnuradio_csis[0]
    return "jpeg"


def _ensure_receiver_running():
    """GNU Radio path delivers decoded latent pieces on UDP; start listener if not already up."""
    global stop_thread, receiver_task_running
    if receiver_task_running:
        return
    stop_thread = False
    receiver_task_running = True
    socketio.start_background_task(receive_pieces)


def _decode_and_emit_latent_for_csis(latent_snapshot: np.ndarray, csis: list[str], is_piece: bool):
    """Run JSCE msg2img for each CSI and push JPEG updates + metrics to the UI."""
    codec = _get_jsce_codec()
    delay = RX_PIECE_EMIT_DELAY_S if is_piece else 0.012
    for sc in csis:
        sc = str(sc)
        try:
            pil_out = codec.msg2img(latent_snapshot, sc)
            pil_out = pil_out.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.LANCZOS)
            frame = np.array(pil_out.convert("RGB"), dtype=np.uint8)
        except Exception as e:
            print(f"JSCE decode failed for csi={sc}: {e}")
            continue

        reconstructed_images[sc] = frame
        orig = original_images.get(sc)
        if orig is not None:
            _update_quality_metrics(sc, frame, orig)
        _emit_metrics(sc)

        img = Image.fromarray(frame.astype("uint8"), "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        socketio.emit(
            "image_update",
            {
                "csi": sc,
                "image": f"data:image/jpeg;base64,{img_base64}",
                "piece": is_piece,
                "preview": is_piece,
            },
        )
        try:
            socketio.sleep(delay)
        except Exception:
            time.sleep(delay)


def receive_pieces():
    global stop_thread
    global reconstructed_images
    global merged_latent
    print("Starting receive_pieces function...")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        print(f"Binding to {HOST}:{PORT}")
        s.bind((HOST, PORT))
        s.settimeout(1.0)  # Add timeout to allow checking stop_thread

        while not stop_thread:
            try:
                data, client_address = s.recvfrom(UDP_RECV_BUF)
                if not data:
                    continue

                kind, obj = _try_decode_udp(data)
                is_piece = kind in ("piece", "piece_csi", "latent_piece")
                if kind == "latent_piece":
                    msg = obj
                    piece = msg["piece"]
                    (yy, xx, cc), _ = piece
                    print(f"Received JSCE latent piece at ({yy}, {xx}, {cc})")

                    stats = _apply_gnuradio_latent_piece(
                        piece,
                        tx_gen=msg.get("tx_gen", 0),
                        expected_patches=msg.get("expected_patches", 0),
                    )

                    csis = list(active_gnuradio_csis) or ["default"]
                    piece_bytes = _estimate_piece_bytes(csis[0], piece)
                    n_patches = stats["n_patches"]
                    n_patches_exp = stats["n_patches_exp"]
                    n_cells = stats["n_cells"]
                    n_cells_exp = stats["n_cells_exp"]

                    for sc in csis:
                        if sc not in metrics_state:
                            init_metrics(sc, expected_pieces=max(n_patches_exp, 1))
                        _update_piece_metrics(sc, piece_bytes)
                        _emit_metrics(sc)

                    if n_patches % 20 == 0 or n_patches == n_patches_exp:
                        print(
                            f"JSCE latent: {n_patches}/{n_patches_exp} patches, "
                            f"{n_cells}/{n_cells_exp} cells"
                        )

                    _maybe_preview_decode_gnuradio_latent(csis, n_cells, n_cells_exp)
                    _maybe_decode_complete_gnuradio_latent(csis)
                elif kind in ("piece", "piece_csi"):
                    if kind == "piece":
                        csi = "default"
                        piece = obj
                    else:
                        csi, piece = obj
                        csi = str(csi)

                    if csi not in reconstructed_images:
                        reconstructed_images[csi] = np.zeros(IMAGE_SIZE, dtype=np.uint8)

                    (yy, xx, cc), _ = piece
                    print(f"Received piece csi={csi} at position ({yy}, {xx}, {cc})")
                    reconstructed_images[csi] = redraw_image(piece, reconstructed_images[csi])
                    img = Image.fromarray(reconstructed_images[csi].astype("uint8"), "RGB")

                    # --- metrics ---
                    if csi not in metrics_state:
                        uph, upw = PIECE_SIZE
                        exp = (IMAGE_SIZE[0] // uph) * (IMAGE_SIZE[1] // upw)
                        init_metrics(csi, expected_pieces=exp)
                    _update_piece_metrics(csi, _estimate_piece_bytes(csi, piece))
                    # periodically compute quality during UDP rebuild
                    st = metrics_state[csi]
                    if st["piece_count"] % 30 == 0:
                        orig = original_images.get(csi)
                        if orig is not None:
                            _update_quality_metrics(csi, reconstructed_images[csi], orig)
                    _emit_metrics(csi)

                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    socketio.emit(
                        "image_update",
                        {
                            "csi": csi,
                            "image": f"data:image/jpeg;base64,{img_base64}",
                            "piece": is_piece,
                        },
                    )
                    delay = RX_PIECE_EMIT_DELAY_S if is_piece else 0.012
                    try:
                        socketio.sleep(delay)
                    except Exception:
                        time.sleep(delay)
                elif kind == "jpeg":
                    img = obj
                    csi = _resolve_gnuradio_rx_csi()
                    if img.size != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
                        img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.LANCZOS)
                    frame = np.array(img.convert("RGB"), dtype=np.uint8)
                    reconstructed_images[csi] = frame
                    print(f"Received GNU Radio decoded image for csi={csi} ({frame.shape[1]}x{frame.shape[0]})")
                    if csi not in metrics_state:
                        init_metrics(csi)
                    st = metrics_state[csi]
                    st["total_bytes"] += frame.nbytes
                    st["_last_piece_time"] = time.time()
                    if st.get("expected_pieces", 0) > 0:
                        st["piece_count"] = st["expected_pieces"]
                    else:
                        st["piece_count"] = max(st["piece_count"], 1)
                    orig = original_images.get(csi)
                    if orig is not None:
                        _update_quality_metrics(csi, frame, orig)
                    _emit_metrics(csi)

                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    socketio.emit(
                        "image_update",
                        {
                            "csi": csi,
                            "image": f"data:image/jpeg;base64,{img_base64}",
                            "piece": is_piece,
                        },
                    )
                    delay = RX_PIECE_EMIT_DELAY_S if is_piece else 0.012
                    try:
                        socketio.sleep(delay)
                    except Exception:
                        time.sleep(delay)
                else:
                    continue

            except socket.timeout:
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                continue


@app.route('/')
def index():
    return render_template(
        "transfer.html",
        default_port=PORT,
        gnuradio_port=GNURADIO_PDU_PORT,
        host=HOST,
        nodes=nodes_state,
        selected_csis=selected_csis,
        channel_states=channel_states,
    )


def _get_jsce_codec():
    """Lazy-load neural codec (torch) for debug bypass path."""
    global _jsce_codec
    if _jsce_codec is not None:
        return _jsce_codec
    with _jsce_lock:
        if _jsce_codec is not None:
            return _jsce_codec
        import torch
        from codec.jsce_codec import JSCE

        if not os.path.isfile(CODEC_CHECKPOINT):
            raise FileNotFoundError(f"JSCE checkpoint missing: {CODEC_CHECKPOINT}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _jsce_codec = JSCE(
            weight_path=CODEC_CHECKPOINT,
            img_size=(CODEC_IMG_SIZE[0], CODEC_IMG_SIZE[1]),
            compressed_channel=CODEC_COMPRESSED_CH,
            device=device,
        )
    return _jsce_codec


def _reset_rx_for_csis(csis: list[str]):
    """Clear server-side rebuild buffers and tell the browser to blank RX tiles (UDP incremental viz)."""
    global reconstructed_images
    for c in csis:
        reconstructed_images.pop(c, None)
        metrics_state.pop(c, None)
        _metrics_last_emit.pop(c, None)
    _reset_merged_latent_state()
    socketio.emit("rx_reset", {"csis": csis})


def _emit_jpeg_piece_update(
        csi: str, frame: np.ndarray, is_piece: bool, delay_s: float, sleep_fn
):
    """Encode frame as JPEG and emit image_update (same shape as receive_pieces)."""
    img = Image.fromarray(frame.astype("uint8"), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    socketio.emit(
        "image_update",
        {
            "csi": csi,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "piece": is_piece,
        },
    )
    sleep_fn(delay_s)


def _stream_neural_decoded_pieces(
        csi: str, decoded_rgb: np.ndarray, piece_delay_s: float, sleep_fn,
        piece_size: Optional[tuple[int, int]] = None,
):
    """
    Shuffled piece-by-piece reveal of one decoded view (mirrors UDP detach/rebuild + channel toy model).
    Runs in a thread when multiple links are active.
    piece_size — (h, w) for each patch; defaults to NEURAL_BYPASS_PIECE_SIZE.
    """
    from image_detach_rebuild import detach_image_patches

    if piece_size is None:
        piece_size = NEURAL_BYPASS_PIECE_SIZE

    csi = str(csi)
    pieces = detach_image_patches(decoded_rgb, piece_size=piece_size)
    buf = reconstructed_images.get(csi)
    if buf is None:
        buf = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        reconstructed_images[csi] = buf
    total = len(pieces)
    for idx, piece in enumerate(pieces):
        if send_stop_flag.is_set():
            return
        if csi not in channel_states:
            _compute_channel_state_for_csi(csi)
        piece2 = _apply_channel_effect(piece, csi)
        if piece2 is None:
            _update_piece_metrics(csi, dropped=True)
            # fill dropped region with zeros and emit so the frontend
            # still updates and the user can see which patches were lost
            (yy, xx, cc), dropped_piece = piece
            dh, dw = dropped_piece.shape[:2]
            buf[yy:yy + dh, xx:xx + dw, :] = 0
            delay = max(RX_PIECE_EMIT_DELAY_S, piece_delay_s)
            _emit_jpeg_piece_update(csi, buf, True, delay, sleep_fn)
            _emit_metrics(csi)
            continue
        _update_piece_metrics(csi, _estimate_piece_bytes(csi, piece2))
        redraw_image(piece2, buf)

        # periodically compute quality metrics so the chart shows a curve
        if idx % 5 == 0 or idx == total - 1:
            orig = original_images.get(csi)
            if orig is not None:
                _update_quality_metrics(csi, buf, orig)

        delay = max(RX_PIECE_EMIT_DELAY_S, piece_delay_s)
        _emit_jpeg_piece_update(csi, buf, True, delay, sleep_fn)
        _emit_metrics(csi)

    # force final metrics emit (bypass throttle) so the client sees 100%
    st = metrics_state.get(csi)
    if st is not None:
        elapsed = time.time() - st["start_time"]
        throughput = _compute_throughput(csi)
        completion = 100.0 if st["expected_pieces"] > 0 else 0.0
        data = {
            "csi": csi,
            "piece_count": st["piece_count"],
            "dropped_count": st["dropped_count"],
            "elapsed_seconds": round(elapsed, 1),
            "throughput_bps": round(throughput, 1),
            "ssim": round(st["last_ssim"], 4),
            "psnr": round(st["last_psnr"], 2),
            "completion_pct": completion,
            "expected_pieces": st["expected_pieces"],
            "time_history": st["time_history"][-100:],
            "ssim_history": [round(v, 4) for v in st["ssim_history"][-100:]],
            "psnr_history": [round(v, 2) for v in st["psnr_history"][-100:]],
        }
        socketio.emit("metrics_update", data)
        _metrics_last_emit[csi] = time.time()


def _neural_bypass_worker(
        saved: list[tuple[str, str]], piece_delay_s: float,
        piece_size: Optional[tuple[int, int]] = None,
):
    """
    JSCE encode/decode without UDP: re-reads images each round (like a looping TX), merges latent,
    decodes per CSI, then streams shuffled patches to the UI so RX rebuilds progressively.
    piece_size — (h, w) for each patch; defaults to NEURAL_BYPASS_PIECE_SIZE.
    """
    if piece_size is None:
        piece_size = NEURAL_BYPASS_PIECE_SIZE

    def _async_sleep(d: float):
        try:
            socketio.sleep(d)
        except Exception:
            time.sleep(d)

    try:
        codec = _get_jsce_codec()
        while not send_stop_flag.is_set():
            image_dict = {}
            for path, csi in saved:
                try:
                    image_dict[str(csi)] = Image.open(path).convert("RGB")
                except Exception as e:
                    socketio.emit(
                        "tx_status",
                        {"message": f"Neural bypass: cannot read {path}: {e}"},
                    )
                    return

            latent = codec.img2msg(image_dict)
            per_csi: dict[str, np.ndarray] = {}
            for _path, csi in saved:
                sc = str(csi)
                pil_out = codec.msg2img(latent, sc)
                pil_out = pil_out.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.LANCZOS)
                per_csi[sc] = np.array(pil_out.convert("RGB"), dtype=np.uint8)

                # --- metrics: init + quality ---
                ph, pw = piece_size
                exp = (IMAGE_SIZE[0] // ph) * (IMAGE_SIZE[1] // pw)
                if sc not in metrics_state:
                    init_metrics(sc, expected_pieces=exp)
                orig = original_images.get(sc)
                if orig is not None:
                    _update_quality_metrics(sc, per_csi[sc], orig)
                _emit_metrics(sc)

            if len(per_csi) == 1:
                (sc, arr) = next(iter(per_csi.items()))
                _stream_neural_decoded_pieces(sc, arr, piece_delay_s, _async_sleep, piece_size=piece_size)
            else:
                threads = []
                for sc, arr in per_csi.items():
                    t = threading.Thread(
                        target=_stream_neural_decoded_pieces,
                        args=(sc, arr, piece_delay_s, time.sleep),
                        kwargs={"piece_size": piece_size},
                        daemon=True,
                    )
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

        socketio.emit(
            "tx_status",
            {"message": f"Neural bypass stopped ({len(saved)} link(s))."},
        )
    except Exception as e:
        socketio.emit("tx_status", {"message": f"Neural bypass failed: {e}"})


def _continual_transmit_worker(
        image_dir: str, csis: list[str], piece_delay_s: float, round_interval_s: float = 3.0,
        hf_dataset_name: str = "", time_delay_mode: bool = False,
        piece_size: Optional[tuple[int, int]] = None,
):
    """
    Continual transmission mode:
    - Each round, sample len(csis) random images from *image_dir* (or *hf_dataset_name*).
    - Encode / decode through JSCE and stream patches to the UI.
    - Per-round originals are set so SSIM/PSNR track fresh images each round.
    - When *time_delay_mode* is True: no chunking; record encode/decode time and emit full image.
    piece_size — (h, w) for each patch; defaults to NEURAL_BYPASS_PIECE_SIZE.
    """
    if piece_size is None:
        piece_size = NEURAL_BYPASS_PIECE_SIZE
    from image_detach_rebuild import detach_image_patches

    def _sleep(d: float):
        try:
            socketio.sleep(d)
        except Exception:
            time.sleep(d)

    # --- HF dataset init (once) ---
    hf_dataset = None
    hf_dataset_size = 0
    if hf_dataset_name:
        try:
            from datasets import load_dataset
            hf_dataset = load_dataset(
                hf_dataset_name,
                cache_dir=HF_DATASET_CACHE_DIR,
                split="train",
            )
            hf_dataset_size = len(hf_dataset)
            socketio.emit(
                "tx_status",
                {"message": f"Loaded HF dataset '{hf_dataset_name}' ({hf_dataset_size} samples)."},
            )
        except Exception as e:
            socketio.emit(
                "tx_status",
                {"message": f"HF dataset load failed: {e}"},
            )
            return

    try:
        codec = _get_jsce_codec()
        round_num = 0
        while not send_stop_flag.is_set():
            round_num += 1

            # --- get source images ---
            if hf_dataset is not None:
                # Sample from HuggingFace dataset
                n_links = len(csis)
                if hf_dataset_size < n_links:
                    socketio.emit(
                        "tx_status",
                        {"message": f"HF dataset has {hf_dataset_size} samples, need {n_links}."},
                    )
                    _sleep(round_interval_s)
                    continue
                indices = random.sample(range(hf_dataset_size), n_links)
                hf_rows = [hf_dataset[i] for i in indices]
                # Determine image key (common keys: "image", "img", "png")
                image_key = "image" if "image" in hf_rows[0] else (next(k for k in ("img", "png") if k in hf_rows[0]))
                selected_files = [f"hf_{i}" for i in indices]
                basenames = [f"hf_{i}" for i in indices]
                pil_images = []
                for row in hf_rows:
                    img = row[image_key]
                    if not isinstance(img, Image.Image):
                        img = Image.open(img).convert("RGB")
                    pil_images.append(img.convert("RGB"))
            else:
                # List valid images from directory
                valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                try:
                    all_files = sorted(
                        f for f in os.listdir(image_dir)
                        if f.lower().endswith(valid_exts)
                    )
                except FileNotFoundError:
                    socketio.emit("tx_status", {"message": f"Directory not found: {image_dir}"})
                    _sleep(round_interval_s)
                    continue

                n_links = len(csis)
                if len(all_files) < n_links:
                    socketio.emit(
                        "tx_status",
                        {"message": f"Round {round_num}: need {n_links} images, found {len(all_files)}. Retrying..."},
                    )
                    _sleep(round_interval_s)
                    continue

                selected = random.sample(all_files, n_links)
                basenames = [os.path.splitext(f)[0] for f in selected]
                selected_files = selected
                pil_images = []
                for f in selected:
                    pil_images.append(Image.open(os.path.join(image_dir, f)).convert("RGB"))

            # --- load originals + build codec input ---
            image_dict: dict[str, Image.Image] = {}
            round_originals: dict[str, np.ndarray] = {}
            for i, csi in enumerate(csis):
                sc = str(csi)
                image_dict[sc] = pil_images[i]
                round_originals[sc] = np.array(
                    pil_images[i].resize((IMAGE_SIZE[1], IMAGE_SIZE[0])), dtype=np.uint8
                )

            # swap originals for quality comparison
            original_images.clear()
            original_images.update(round_originals)

            socketio.emit("round_update", {"round": round_num, "files": basenames, "csis": list(csis)})
            socketio.emit(
                "tx_status",
                {"message": f"Round {round_num}/∞: {', '.join(basenames)}"},
            )

            # --- encode merged latent (timed) ---
            t0 = time.perf_counter()
            latent = codec.img2msg(image_dict)
            encode_time = time.perf_counter() - t0

            # decode per CSI (timed individually for time delay mode)
            per_csi: dict[str, np.ndarray] = {}
            decode_times: dict[str, float] = {}
            for csi in csis:
                sc = str(csi)
                t0 = time.perf_counter()
                pil_out = codec.msg2img(latent, sc)
                decode_times[sc] = time.perf_counter() - t0
                pil_out = pil_out.resize(
                    (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.LANCZOS
                )
                per_csi[sc] = np.array(pil_out.convert("RGB"), dtype=np.uint8)

            # stream patches (parallel for multiple links)
            if n_links == 1:
                sc = csis[0]
                ph, pw = piece_size
                init_metrics(sc, expected_pieces=(IMAGE_SIZE[0] // ph) * (IMAGE_SIZE[1] // pw))
                _stream_neural_decoded_pieces(sc, per_csi[sc], piece_delay_s, _sleep, piece_size=piece_size)
            else:
                threads = []
                for sc in csis:
                    ph, pw = piece_size
                    init_metrics(sc, expected_pieces=(IMAGE_SIZE[0] // ph) * (IMAGE_SIZE[1] // pw))
                    t = threading.Thread(
                        target=_stream_neural_decoded_pieces,
                        args=(sc, per_csi[sc], piece_delay_s, time.sleep),
                        kwargs={"piece_size": piece_size},
                        daemon=True,
                    )
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

            # time-delay recording (if enabled — alongside normal chunking)
            if time_delay_mode:
                for sc in csis:
                    ph, pw = piece_size
                    expected = (IMAGE_SIZE[0] // ph) * (IMAGE_SIZE[1] // pw)
                    total_time = encode_time + decode_times[sc]
                    per_chunk_avg = total_time / expected if expected > 0 else total_time

                    time_delay_data.setdefault(sc, []).append({
                        "round": round_num,
                        "encode_time": round(encode_time, 4),
                        "decode_time": round(decode_times[sc], 4),
                        "total_time": round(total_time, 4),
                        "per_chunk_avg": round(per_chunk_avg, 6),
                    })

                    history = time_delay_data.get(sc, [])
                    mean_per_chunk = np.mean([h["per_chunk_avg"] for h in history]) if history else 0
                    socketio.emit("time_delay_update", {
                        "csi": sc,
                        "round": round_num,
                        "encode_time": round(encode_time, 4),
                        "decode_time": round(decode_times[sc], 4),
                        "total_time": round(total_time, 4),
                        "per_chunk_avg": round(per_chunk_avg, 6),
                        "mean_per_chunk_avg": round(mean_per_chunk, 6),
                        "history": history[-200:],
                    })

            socketio.emit(
                "tx_status",
                {
                    "message": f"Round {round_num} done{', encode {:.3f}s'.format(encode_time) if time_delay_mode else ''} — {round_interval_s:.0f}s pause..."},
            )

            # pause between rounds
            steps = int(round_interval_s / 0.1)
            for _ in range(steps):
                if send_stop_flag.is_set():
                    return
                _sleep(0.1)

    except Exception as e:
        socketio.emit("tx_status", {"message": f"Continual tx failed: {e}"})
        import traceback
        traceback.print_exc()


def _transmit_jsce_latent_over_gnuradio(
        sock: socket.socket,
        image_dict: dict[str, Image.Image],
        csis: list[str],
        gr_port: int,
        piece_delay_s: float,
        piece_size: tuple[int, int],
) -> bool:
    """
    JSCE img2msg → spatial latent atoms (all channels, 802.11-safe) → UDP to GNU Radio.
    Returns False on MTU / codec failure.
    """
    from image_detach_rebuild import detach_image

    try:
        codec = _get_jsce_codec()
        latent = codec.img2msg(image_dict)
    except Exception as e:
        socketio.emit("tx_status", {"message": f"JSCE encode failed: {e}"})
        return False

    pieces = detach_image(latent, piece_size=piece_size)
    exp = _expected_latent_pieces(piece_size)
    pkt_bytes = _estimate_gnuradio_latent_piece_packet_bytes(piece_size[0], piece_size[1])

    global _gnuradio_latent_tx_generation
    _gnuradio_latent_tx_generation += 1
    tx_gen = _gnuradio_latent_tx_generation

    socketio.emit(
        "tx_status",
        {
            "message": (
                f"JSCE latent TX round {tx_gen}: patch {piece_size[0]}x{piece_size[1]}x{CODEC_COMPRESSED_CH} "
                f"(float16 wire, ~{pkt_bytes} B/frame, {exp} patches/round)"
            ),
        },
    )
    for sc in csis:
        init_metrics(str(sc), expected_pieces=exp)

    channel_csi = str(csis[0]) if csis else "3-4"
    for piece in pieces:
        if send_stop_flag.is_set():
            break
        if channel_csi not in channel_states:
            _compute_channel_state_for_csi(channel_csi)
        piece2 = _apply_channel_effect_latent(piece, channel_csi)
        if piece2 is None:
            for sc in csis:
                _update_piece_metrics(str(sc), dropped=True)
                _emit_metrics(str(sc))
            continue

        packet = _len_prefixed_packet(
            _pickle_jsce_latent_payload(piece2, tx_gen=tx_gen, expected_patches=exp)
        )
        if len(packet) > GNURADIO_MAC_MSDU_MAX:
            socketio.emit(
                "tx_status",
                {
                    "message": _gnuradio_latent_piece_size_error(piece_size)
                    or f"Latent packet {len(packet)} B exceeds 802.11 MAC limit {GNURADIO_MAC_MSDU_MAX} B",
                },
            )
            return False
        sock.sendto(packet, (HOST, int(gr_port)))
        time.sleep(piece_delay_s)
    return True


def _continual_gnuradio_worker(
        image_dir: str, csis: list[str], piece_delay_s: float, round_interval_s: float = 3.0,
        hf_dataset_name: str = "", gr_port: int = GNURADIO_PDU_PORT,
        piece_size=None,
):
    """
    Continual transmission via GNU Radio + JSCE:
    - Each round, sample one image per CSI link.
    - img2msg merges latents, detach spatial patches, send over UDP to GNU Radio.
    - RX rebuilds merged latent and msg2img per CSI (port 10010).
    """
    if piece_size is None:
        piece_size = _resolve_gnuradio_latent_piece_size()
    else:
        piece_size = _resolve_gnuradio_latent_piece_size(user_hint=piece_size)

    def _sleep(d: float):
        try:
            socketio.sleep(d)
        except Exception:
            time.sleep(d)

    # --- HF dataset init (once) ---
    hf_dataset = None
    hf_dataset_size = 0
    if hf_dataset_name:
        try:
            from datasets import load_dataset
            hf_dataset = load_dataset(
                hf_dataset_name,
                cache_dir=HF_DATASET_CACHE_DIR,
                split="train",
            )
            hf_dataset_size = len(hf_dataset)
            socketio.emit(
                "tx_status",
                {"message": f"Loaded HF dataset '{hf_dataset_name}' ({hf_dataset_size} samples)."},
            )
        except Exception as e:
            socketio.emit("tx_status", {"message": f"HF dataset load failed: {e}"})
            return

    try:
        round_num = 0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            while not send_stop_flag.is_set():
                round_num += 1

                # --- get source images ---
                if hf_dataset is not None:
                    n_links = len(csis)
                    if hf_dataset_size < n_links:
                        socketio.emit(
                            "tx_status",
                            {"message": f"HF dataset has {hf_dataset_size} samples, need {n_links}."},
                        )
                        _sleep(round_interval_s)
                        continue
                    indices = random.sample(range(hf_dataset_size), n_links)
                    hf_rows = [hf_dataset[i] for i in indices]
                    image_key = "image" if "image" in hf_rows[0] else (
                        next(k for k in ("img", "png") if k in hf_rows[0]))
                    basenames = [f"hf_{i}" for i in indices]
                    pil_images = []
                    for row in hf_rows:
                        img = row[image_key]
                        if not isinstance(img, Image.Image):
                            img = Image.open(img).convert("RGB")
                        pil_images.append(img.convert("RGB"))
                else:
                    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                    try:
                        all_files = sorted(
                            f for f in os.listdir(image_dir)
                            if f.lower().endswith(valid_exts)
                        )
                    except FileNotFoundError:
                        socketio.emit("tx_status", {"message": f"Directory not found: {image_dir}"})
                        _sleep(round_interval_s)
                        continue

                    n_links = len(csis)
                    if len(all_files) < n_links:
                        socketio.emit(
                            "tx_status",
                            {
                                "message": f"Round {round_num}: need {n_links} images, found {len(all_files)}. Retrying..."},
                        )
                        _sleep(round_interval_s)
                        continue

                    selected = random.sample(all_files, n_links)
                    basenames = [os.path.splitext(f)[0] for f in selected]
                    pil_images = []
                    for f in selected:
                        pil_images.append(Image.open(os.path.join(image_dir, f)).convert("RGB"))

                # --- originals for metrics + JSCE encode input ---
                round_originals = {}
                image_dict: dict[str, Image.Image] = {}
                for i, csi in enumerate(csis):
                    sc = str(csi)
                    pil_images[i].load()
                    image_dict[sc] = pil_images[i].convert("RGB")
                    round_originals[sc] = np.array(
                        pil_images[i].resize((IMAGE_SIZE[1], IMAGE_SIZE[0])), dtype=np.uint8
                    )
                original_images.clear()
                original_images.update(round_originals)

                socketio.emit("round_update", {"round": round_num, "files": basenames, "csis": list(csis)})
                socketio.emit(
                    "tx_status",
                    {
                        "message": (
                            f"Round {round_num}/inf via GNU Radio+JSCE: {', '.join(basenames)} "
                            f"(latent patch {piece_size[0]}x{piece_size[1]}, port {gr_port})"
                        ),
                    },
                )

                if not _transmit_jsce_latent_over_gnuradio(
                    s, image_dict, csis, gr_port, piece_delay_s, piece_size,
                ):
                    return

                socketio.emit(
                    "tx_status",
                    {"message": f"Round {round_num} done - {round_interval_s:.0f}s pause..."},
                )

                steps = int(round_interval_s / 0.1)
                for _ in range(steps):
                    if send_stop_flag.is_set():
                        return
                    _sleep(0.1)

    except Exception as e:
        socketio.emit("tx_status", {"message": f"Continual GNU Radio tx failed: {e}"})
        import traceback
        traceback.print_exc()


def _gnuradio_jsce_user_worker(
        saved: list[tuple[str, str]],
        gr_port: int,
        piece_delay_s: float,
        piece_size: Optional[tuple[int, int]] = None,
):
    """
    User-defined mode: re-read uploaded images each round, JSCE encode merged latent,
    send patches to GNU Radio until stopped.
    """
    if piece_size is None:
        piece_size = _resolve_gnuradio_latent_piece_size()
    else:
        piece_size = _resolve_gnuradio_latent_piece_size(user_hint=piece_size)
    csis = [str(csi) for _, csi in saved]

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while not send_stop_flag.is_set():
            image_dict: dict[str, Image.Image] = {}
            try:
                for path, csi in saved:
                    image_dict[str(csi)] = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"GNU Radio JSCE re-read failed: {e}")
                time.sleep(0.2)
                continue

            if not _transmit_jsce_latent_over_gnuradio(
                s, image_dict, csis, gr_port, piece_delay_s, piece_size,
            ):
                return


@app.route("/send_image", methods=["POST"])
def handle_send_image():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    mode = (request.form.get("mode") or "user").strip()

    global _sim_channel_effect_enabled
    _sim_channel_effect_enabled = _form_flag("sim_channel_effect")
    sim_ch_label = "sim channel ON" if _sim_channel_effect_enabled else "sim channel OFF"

    # --- Continual transmission mode ---
    if mode == "continual":
        image_dir = (request.form.get("image_dir") or "").strip()
        hf_dataset_name = (request.form.get("hf_dataset_name") or "").strip()
        use_hf = (request.form.get("use_hf_dataset") or "").strip() in ("1", "on", "true", "yes")
        time_delay_mode = (request.form.get("time_delay_switch") or "").strip() in ("1", "on", "true", "yes")
        neural_bypass = (request.form.get("neural_bypass") or "").strip() in ("1", "on", "true", "yes")

        if use_hf and hf_dataset_name:
            # HF dataset mode: image_dir is optional
            pass
        elif not image_dir or not os.path.isdir(image_dir):
            return jsonify({"status": "error", "reason": f"Invalid image directory: {image_dir}"}), 400

        try:
            round_interval_s = float(request.form.get("round_interval") or 3)
        except ValueError:
            round_interval_s = 3.0
        round_interval_s = max(1.0, min(30.0, round_interval_s))

        try:
            packet_loss_pct = float(request.form.get("packet_loss_pct") or 0)
        except ValueError:
            packet_loss_pct = 0.0
        extra_loss = packet_loss_pct / 100.0

        try:
            piece_delay_ms = float(request.form.get("piece_delay_ms") or 25)
        except ValueError:
            piece_delay_ms = 25.0
        piece_delay_s = max(5.0, min(200.0, piece_delay_ms)) / 1000.0

        try:
            bypass_piece_size = int(request.form.get("bypass_piece_size") or 50)
        except ValueError:
            bypass_piece_size = 50
        bypass_piece_size = max(10, min(300, bypass_piece_size))

        csis = list(selected_csis) if selected_csis else ["3-4"]
        for csi in csis:
            channel_states.setdefault(csi, {})["drop_prob_extra"] = extra_loss
            _compute_channel_state_for_csi(csi)

        try:
            gr_port = int(request.form.get("gnuradio_port") or GNURADIO_PDU_PORT)
        except ValueError:
            gr_port = GNURADIO_PDU_PORT

        _reset_rx_for_csis(csis)
        # Reset time delay data when starting
        time_delay_data.clear()
        socketio.emit(
            "canvas_state",
            {"nodes": nodes_state, "selected_csis": csis, "channel_states": channel_states},
        )
        send_stop_flag.clear()
        if neural_bypass:
            socketio.start_background_task(
                _continual_transmit_worker, image_dir, csis, piece_delay_s, round_interval_s,
                hf_dataset_name if use_hf else "", time_delay_mode,
                piece_size=(bypass_piece_size, bypass_piece_size),
            )
            links_info = f"HF dataset '{hf_dataset_name}'" if use_hf else f"dir '{image_dir}'"
            mode_info = " (time delay viz)" if time_delay_mode else ""
            socketio.emit(
                "tx_status",
                {
                    "message": (
                        f"Continual TX (neural bypass) from {links_info} ({len(csis)} link(s), "
                        f"~{round_interval_s:.0f}s interval{mode_info}). Stop to end."
                    )
                },
            )
        else:
            latent_piece_size = _resolve_gnuradio_latent_piece_size()
            mtu_err = _gnuradio_latent_piece_size_error(latent_piece_size)
            if mtu_err:
                return jsonify({"status": "error", "reason": mtu_err}), 400
            pkt_b = _estimate_gnuradio_latent_piece_packet_bytes(*latent_piece_size)
            _reset_gnuradio_rx_state(csis)
            _ensure_receiver_running()
            socketio.start_background_task(
                _continual_gnuradio_worker, image_dir, csis, piece_delay_s, round_interval_s,
                hf_dataset_name if use_hf else "", gr_port,
                piece_size=latent_piece_size,
            )
            links_info = f"HF dataset '{hf_dataset_name}'" if use_hf else f"dir '{image_dir}'"
            socketio.emit(
                "tx_status",
                {
                    "message": (
                        f"Continual TX via GNU Radio+JSCE (port {gr_port}) from {links_info} "
                        f"({len(csis)} link(s), patch {latent_piece_size[0]}x{latent_piece_size[1]} "
                        f"float16 ~{pkt_b} B/frame, preview decodes every ~10%, "
                        f"~{round_interval_s:.0f}s interval, {sim_ch_label}). Stop to end."
                    )
                },
            )
        return jsonify({"status": "sending", "links": len(csis), "mode": "continual"})

    # --- User-defined mode (original) ---
    neural_bypass = (request.form.get("neural_bypass") or "").strip() in ("1", "on", "true", "yes")
    if not neural_bypass:
        try:
            gr_port = int(request.form.get("gnuradio_port") or GNURADIO_PDU_PORT)
        except ValueError:
            gr_port = GNURADIO_PDU_PORT
        legacy_port = request.form.get("port")
        if legacy_port:
            try:
                gr_port = int(legacy_port)
            except ValueError:
                pass

    try:
        packet_loss_pct = float(request.form.get("packet_loss_pct") or 0)
    except ValueError:
        packet_loss_pct = 0.0
    packet_loss_pct = max(0.0, min(90.0, packet_loss_pct))
    extra_loss = packet_loss_pct / 100.0

    try:
        piece_delay_ms = float(request.form.get("piece_delay_ms") or 25)
    except ValueError:
        piece_delay_ms = 25.0
    piece_delay_ms = max(5.0, min(200.0, piece_delay_ms))
    piece_delay_s = piece_delay_ms / 1000.0

    try:
        bypass_piece_size = int(request.form.get("bypass_piece_size") or 50)
    except ValueError:
        bypass_piece_size = 50
    bypass_piece_size = max(10, min(150, bypass_piece_size))

    # multi-link: file_0 + csi_0, file_1 + csi_1, ...
    to_send: list[tuple[str, str]] = []
    for key in request.form.keys():
        if not key.startswith("csi_"):
            continue
        idx = key.split("_", 1)[1]
        csi = (request.form.get(key) or "").strip()
        f = request.files.get(f"file_{idx}")
        if csi and f:
            to_send.append((csi, idx))

    # legacy single-file fallback
    if not to_send:
        f = request.files.get("file")
        if not f:
            return jsonify({"status": "error", "reason": "missing file(s)"}), 400
        csi = (request.form.get("csi") or "").strip() or (selected_csis[0] if selected_csis else "3-4")
        to_send = [(csi, "__single__")]

    saved: list[tuple[str, str]] = []
    for csi, idx in to_send:
        f = request.files.get("file") if idx == "__single__" else request.files.get(f"file_{idx}")
        if not f:
            continue
        file_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(file_path)
        channel_states.setdefault(csi, {})["drop_prob_extra"] = extra_loss
        _compute_channel_state_for_csi(csi)
        # save original for SSIM / PSNR comparison
        try:
            orig = np.array(Image.open(file_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0])), dtype=np.uint8)
            original_images[str(csi)] = orig
        except Exception:
            original_images.pop(str(csi), None)
        saved.append((file_path, csi))

    started = len(saved)
    if started == 0:
        return jsonify({"status": "error", "reason": "no files saved"}), 400

    rx_csis = [csi for _, csi in saved]
    _reset_rx_for_csis(rx_csis)
    socketio.emit(
        "canvas_state",
        {"nodes": nodes_state, "selected_csis": selected_csis, "channel_states": channel_states},
    )

    if neural_bypass:
        send_stop_flag.clear()
        socketio.start_background_task(
            _neural_bypass_worker, saved, piece_delay_s,
            piece_size=(bypass_piece_size, bypass_piece_size),
        )
        socketio.emit(
            "tx_status",
            {
                "message": (
                    f"Neural bypass: JSCE + chunked Socket.IO for {started} link(s); "
                    f"Stop to end. Re-saves / disk edits apply on the next full round."
                )
            },
        )
        return jsonify({"status": "sending", "links": started, "mode": "neural"})

    latent_piece_size = _resolve_gnuradio_latent_piece_size()
    mtu_err = _gnuradio_latent_piece_size_error(latent_piece_size)
    if mtu_err:
        return jsonify({"status": "error", "reason": mtu_err}), 400
    pkt_b = _estimate_gnuradio_latent_piece_packet_bytes(*latent_piece_size)

    _reset_gnuradio_rx_state(rx_csis)
    _ensure_receiver_running()
    send_stop_flag.clear()
    socketio.start_background_task(
        _gnuradio_jsce_user_worker,
        saved,
        gr_port,
        piece_delay_s,
        latent_piece_size,
    )

    socketio.emit(
        "tx_status",
        {
            "message": (
                f"GNU Radio+JSCE TX: {started} link(s) → {HOST}:{gr_port} "
                f"(patch {latent_piece_size[0]}x{latent_piece_size[1]} float16 ~{pkt_b} B/frame, "
                f"preview every ~10%, gap {piece_delay_ms:.0f} ms, loss {packet_loss_pct:.0f}%, "
                f"{sim_ch_label}). "
                f"Stop to end."
            )
        },
    )
    return jsonify({"status": "sending", "links": started, "mode": "udp"})


@app.route("/stop_send", methods=["POST"])
def handle_stop_send():
    send_stop_flag.set()
    socketio.emit("tx_status", {"message": "TX stopped"})
    return jsonify({"status": "stopped"})


@socketio.on('connect')
def handle_connect():
    print("Client connected")  # Debug log
    for csi in selected_csis:
        _compute_channel_state_for_csi(csi)
    socketio.emit("canvas_state",
                  {"nodes": nodes_state, "selected_csis": selected_csis, "channel_states": channel_states})


@socketio.on('start_receiving')
def handle_start():
    global stop_thread, receiver_task_running
    print("Received start signal")  # Debug log
    stop_thread = False
    if not receiver_task_running:
        receiver_task_running = True
        socketio.start_background_task(receive_pieces)
    """thread = threading.Thread(target=receive_pieces)
    thread.daemon = True
    thread.start()"""


@socketio.on('stop_receiving')
def handle_stop():
    global stop_thread, receiver_task_running
    print("Received stop signal")  # Debug log
    stop_thread = True
    receiver_task_running = False


@socketio.on("canvas_update")
def handle_canvas_update(data):
    """
    data: { nodes: [...], selected_csis: ["13-14", ...] }
    """
    global nodes_state, selected_csis
    try:
        nodes = data.get("nodes", [])
        csis = data.get("selected_csis", [])
        if isinstance(nodes, list) and nodes:
            nodes_state = nodes
        if isinstance(csis, list):
            selected_csis = [str(x) for x in csis if str(x)]
        for csi in selected_csis:
            _compute_channel_state_for_csi(csi)
        socketio.emit("canvas_state",
                      {"nodes": nodes_state, "selected_csis": selected_csis, "channel_states": channel_states})
    except Exception as e:
        socketio.emit("status", {"message": f"canvas_update error: {e}"})


if __name__ == "__main__":
    # debug=True but no reloader: stat reloader often restarts when site-packages
    # (e.g. transformers) is touched, dropping all Socket.IO clients.
    socketio.run(
        app,
        host=HOST,
        port=5000,
        debug=True,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )