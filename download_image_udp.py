try:
    import eventlet  # type: ignore
    eventlet.monkey_patch()
    _ASYNC_MODE = "eventlet"
except Exception:
    eventlet = None
    _ASYNC_MODE = "threading"

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
# multi-link reconstruction (keyed by CSI like "13-14")
reconstructed_images: dict[str, np.ndarray] = {}

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
# Slightly slower Socket.IO emit after each UDP piece so the browser can paint incremental rebuilds.
RX_PIECE_EMIT_DELAY_S = 0.03
UDP_RECV_BUF = 65535


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


def _try_decode_udp(data: bytes):
    """
    Returns one of:
      ("piece", ((y,x,c), piece_array))
      ("jpeg", pil_image_rgb)
      ("none", None)
    """
    payload = _safe_unpack_len_prefixed(data)

    # 1) pickle piece
    try:
        obj = pickle.loads(payload)

        # multi-link formats:
        #  - {"csi": "13-14", "piece": ((y,x,c), arr)}
        #  - ("13-14", ((y,x,c), arr))
        if isinstance(obj, dict) and "piece" in obj and "csi" in obj:
            csi = str(obj.get("csi"))
            piece = obj.get("piece")
            if isinstance(piece, tuple) and len(piece) == 2 and isinstance(piece[0], tuple) and len(piece[0]) == 3:
                return ("piece_csi", (csi, piece))
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str):
            csi = obj[0]
            piece = obj[1]
            if isinstance(piece, tuple) and len(piece) == 2 and isinstance(piece[0], tuple) and len(piece[0]) == 3:
                return ("piece_csi", (csi, piece))

        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], tuple) and len(obj[0]) == 3:
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

def _apply_channel_effect(piece, csi: str):
    """
    Apply a lightweight, visually-obvious "channel" effect to a piece:
    - random drop (piece not delivered — caller should skip update)
    - additive gaussian noise scaled by SNR

    Drop is decided independently on each transmission attempt so the same
    spatial patch is not permanently lost due to a fixed (y, x) seed.
    Returns None when the piece is dropped.
    """
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


def receive_pieces():
    global stop_thread
    global reconstructed_images
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
                is_piece = kind in ("piece", "piece_csi")
                if is_piece:
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
                        exp = (IMAGE_SIZE[0] // uph) * (IMAGE_SIZE[1] // upw) * 3
                        init_metrics(csi, expected_pieces=exp)
                    _update_piece_metrics(csi, _estimate_piece_bytes(csi, piece))
                    # periodically compute quality during UDP rebuild
                    st = metrics_state[csi]
                    if st["piece_count"] % 30 == 0:
                        orig = original_images.get(csi)
                        if orig is not None:
                            _update_quality_metrics(csi, reconstructed_images[csi], orig)
                    _emit_metrics(csi)
                elif kind == "jpeg":
                    img = obj
                    csi = "jpeg"
                else:
                    continue

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
                {"message": f"Round {round_num} done{', encode {:.3f}s'.format(encode_time) if time_delay_mode else ''} — {round_interval_s:.0f}s pause..."},
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


def _continual_gnuradio_worker(
    image_dir: str, csis: list[str], piece_delay_s: float, round_interval_s: float = 3.0,
    hf_dataset_name: str = "", gr_port: int = GNURADIO_PDU_PORT,
    piece_size=(50, 50),
):
    """
    Continual transmission mode via GNU Radio:
    - Each round, sample images from *image_dir* (or *hf_dataset_name*).
    - Detach each image into pieces and send via UDP to GNU Radio PDU port.
    - Pieces are tagged with CSI so the receiver (port 10010) can route them.
    - Toy channel model (drop + noise) applied server-side.
    """
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
                    image_key = "image" if "image" in hf_rows[0] else (next(k for k in ("img", "png") if k in hf_rows[0]))
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
                            {"message": f"Round {round_num}: need {n_links} images, found {len(all_files)}. Retrying..."},
                        )
                        _sleep(round_interval_s)
                        continue

                    selected = random.sample(all_files, n_links)
                    basenames = [os.path.splitext(f)[0] for f in selected]
                    pil_images = []
                    for f in selected:
                        pil_images.append(Image.open(os.path.join(image_dir, f)).convert("RGB"))

                # --- load originals ---
                round_originals = {}
                for i, csi in enumerate(csis):
                    sc = str(csi)
                    round_originals[sc] = np.array(
                        pil_images[i].resize((IMAGE_SIZE[1], IMAGE_SIZE[0])), dtype=np.uint8
                    )
                original_images.clear()
                original_images.update(round_originals)

                socketio.emit("round_update", {"round": round_num, "files": basenames, "csis": list(csis)})
                socketio.emit(
                    "tx_status",
                    {"message": f"Round {round_num}/inf via GNU Radio: {', '.join(basenames)} (port {gr_port})"},
                )

                # --- detach each image and send pieces via UDP ---
                for i, csi in enumerate(csis):
                    sc = str(csi)
                    img_arr = np.array(pil_images[i].resize((IMAGE_SIZE[1], IMAGE_SIZE[0])), dtype=np.uint8)
                    pieces = detach_image_patches(img_arr, piece_size=piece_size)

                    if sc not in metrics_state:
                        ph, pw = piece_size
                        exp = (IMAGE_SIZE[0] // ph) * (IMAGE_SIZE[1] // pw) * 3
                        init_metrics(sc, expected_pieces=exp)

                    for piece in pieces:
                        if send_stop_flag.is_set():
                            break
                        if sc not in channel_states:
                            _compute_channel_state_for_csi(sc)
                        piece2 = _apply_channel_effect(piece, sc)
                        if piece2 is None:
                            _update_piece_metrics(sc, dropped=True)
                            _emit_metrics(sc)
                            continue

                        payload = pickle.dumps({"csi": sc, "piece": piece2}, protocol=pickle.HIGHEST_PROTOCOL)
                        message_size = struct.pack("=L", len(payload))
                        s.sendto(message_size + payload, (HOST, int(gr_port)))
                        _update_piece_metrics(sc, _estimate_piece_bytes(sc, piece2))
                        _emit_metrics(sc)
                        time.sleep(piece_delay_s)

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



def _send_image_worker(image_path: str, port: int, csi: str, piece_delay_s: float):
    """
    Send detached image pieces via UDP.
    Note: This uses the same "piece" structure as `image_detach_rebuild.detach_image`.
    """
    from image_detach_rebuild import detach_image

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while not send_stop_flag.is_set():
            try:
                original_image = np.array(
                    Image.open(image_path).convert("RGB").resize((300, 300)),
                    dtype=np.uint8,
                )
            except Exception as e:
                print(f"UDP TX re-read failed {image_path}: {e}")
                time.sleep(0.2)
                continue
            pieces = detach_image(original_image)
            for piece in pieces:
                if send_stop_flag.is_set():
                    break

                if csi not in channel_states:
                    _compute_channel_state_for_csi(csi)

                piece2 = _apply_channel_effect(piece, csi)
                if piece2 is None:
                    continue
                payload = pickle.dumps({"csi": csi, "piece": piece2}, protocol=pickle.HIGHEST_PROTOCOL)
                message_size = struct.pack("=L", len(payload))
                s.sendto(message_size + payload, (HOST, int(port)))
                time.sleep(piece_delay_s)


@app.route("/send_image", methods=["POST"])
def handle_send_image():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    mode = (request.form.get("mode") or "user").strip()

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
            socketio.start_background_task(
                _continual_gnuradio_worker, image_dir, csis, piece_delay_s, round_interval_s,
                hf_dataset_name if use_hf else "", gr_port,
                piece_size=(bypass_piece_size, bypass_piece_size),
            )
            links_info = f"HF dataset '{hf_dataset_name}'" if use_hf else f"dir '{image_dir}'"
            socketio.emit(
                "tx_status",
                {
                    "message": (
                        f"Continual TX via GNU Radio (port {gr_port}) from {links_info} "
                        f"({len(csis)} link(s), ~{round_interval_s:.0f}s interval). Stop to end."
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

    send_stop_flag.clear()
    for file_path, csi in saved:
        threading.Thread(
            target=_send_image_worker,
            args=(file_path, gr_port, csi, piece_delay_s),
            daemon=True,
        ).start()

    socketio.emit(
        "tx_status",
        {
            "message": (
                f"UDP TX: {started} link(s) → {HOST}:{gr_port} "
                f"(piece gap {piece_delay_ms:.0f} ms, extra loss {packet_loss_pct:.0f}%)"
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
    socketio.emit("canvas_state", {"nodes": nodes_state, "selected_csis": selected_csis, "channel_states": channel_states})

@socketio.on('start_receiving')
def handle_start():
    global stop_thread
    print("Received start signal")  # Debug log
    stop_thread = False
    socketio.start_background_task(receive_pieces)
    """thread = threading.Thread(target=receive_pieces)
    thread.daemon = True
    thread.start()"""

@socketio.on('stop_receiving')
def handle_stop():
    global stop_thread
    print("Received stop signal")  # Debug log
    stop_thread = True


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
        socketio.emit("canvas_state", {"nodes": nodes_state, "selected_csis": selected_csis, "channel_states": channel_states})
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