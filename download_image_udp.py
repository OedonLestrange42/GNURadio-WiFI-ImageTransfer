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
from image_detach_rebuild import redraw_image, PATCH_PIECE_SIZE as NEURAL_BYPASS_PIECE_SIZE
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from PIL import Image
import base64
import io
import os
import threading
import time
import random

# Configuration
HOST = 'localhost'
PORT = 10010
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=_ASYNC_MODE)
stop_thread = False  # receiver stop flag
# multi-link reconstruction (keyed by CSI like "13-14")
reconstructed_images: dict[str, np.ndarray] = {}

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
    csi: str, decoded_rgb: np.ndarray, piece_delay_s: float, sleep_fn
):
    """
    Shuffled piece-by-piece reveal of one decoded view (mirrors UDP detach/rebuild + channel toy model).
    Runs in a thread when multiple links are active.
    """
    from image_detach_rebuild import detach_image_patches

    csi = str(csi)
    pieces = detach_image_patches(decoded_rgb, piece_size=NEURAL_BYPASS_PIECE_SIZE)
    buf = reconstructed_images.get(csi)
    if buf is None:
        buf = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        reconstructed_images[csi] = buf
    for piece in pieces:
        if send_stop_flag.is_set():
            return
        if csi not in channel_states:
            _compute_channel_state_for_csi(csi)
        piece2 = _apply_channel_effect(piece, csi)
        if piece2 is None:
            continue
        redraw_image(piece2, buf)
        delay = max(RX_PIECE_EMIT_DELAY_S, piece_delay_s)
        _emit_jpeg_piece_update(csi, buf, True, delay, sleep_fn)


def _neural_bypass_worker(saved: list[tuple[str, str]], piece_delay_s: float):
    """
    JSCE encode/decode without UDP: re-reads images each round (like a looping TX), merges latent,
    decodes per CSI, then streams shuffled patches to the UI so RX rebuilds progressively.
    """
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

            if len(per_csi) == 1:
                (sc, arr) = next(iter(per_csi.items()))
                _stream_neural_decoded_pieces(sc, arr, piece_delay_s, _async_sleep)
            else:
                threads = []
                for sc, arr in per_csi.items():
                    t = threading.Thread(
                        target=_stream_neural_decoded_pieces,
                        args=(sc, arr, piece_delay_s, time.sleep),
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

    neural_bypass = (request.form.get("neural_bypass") or "").strip() in ("1", "on", "true", "yes")
    port = request.form.get("port")
    if not neural_bypass:
        if not port:
            return jsonify({"status": "error", "reason": "missing port"}), 400
        try:
            port_i = int(port)
        except ValueError:
            return jsonify({"status": "error", "reason": "invalid port"}), 400
    else:
        try:
            port_i = int(port) if port else PORT
        except ValueError:
            port_i = PORT

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
        socketio.start_background_task(_neural_bypass_worker, saved, piece_delay_s)
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
            args=(file_path, port_i, csi, piece_delay_s),
            daemon=True,
        ).start()

    socketio.emit(
        "tx_status",
        {
            "message": (
                f"UDP TX: {started} link(s) → {HOST}:{port_i} "
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