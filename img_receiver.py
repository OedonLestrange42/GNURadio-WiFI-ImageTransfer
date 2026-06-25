import struct

import pmt
import socket
from gnuradio import gr


class blk(gr.sync_block):
    """Forward decoded MAC MSDU image bytes to download_image_udp on localhost:10010."""

    MAC_HEADER_LEN = 24
    RX_HOST = "localhost"
    RX_PORT = 10010

    def __init__(self, image_hight: int = 300, image_width: int = 300):
        gr.basic_block.__init__(
            self,
            name="Extract Pics",
            in_sig=None,
            out_sig=None,
        )
        self.message_port_register_in(pmt.intern("MAC"))
        self.set_msg_handler(pmt.intern("MAC"), self.handle_msg)
        self.skt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.H = image_hight
        self.W = image_width

    @staticmethod
    def _to_bytes(data) -> bytes:
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, str):
            return data.encode("latin1")
        return bytes(data)

    def _extract_image_payload(self, data: bytes) -> bytes:
        if len(data) < self.MAC_HEADER_LEN + 4:
            raise ValueError(
                f"frame too short ({len(data)} B); need at least "
                f"{self.MAC_HEADER_LEN + 4} B (MAC header + length prefix)"
            )
        target = data[self.MAC_HEADER_LEN:]
        (payload_len,) = struct.unpack("=L", target[:4])
        payload = target[4:]
        if 0 < payload_len <= len(payload):
            return payload[:payload_len]
        return payload

    def handle_msg(self, msg):
        try:
            parsed = pmt.to_python(msg)
            if isinstance(parsed, (tuple, list)):
                data = parsed[-1]
            else:
                data = parsed
            raw = self._to_bytes(data)
            img = self._extract_image_payload(raw)
            if not img:
                print("Extract Pics: empty image payload after header strip")
                return
            self.skt.sendto(img, (self.RX_HOST, self.RX_PORT))
            print(f"Extract Pics: forwarded {len(img)} B to {self.RX_HOST}:{self.RX_PORT}")
        except Exception as e:
            print(f"Extract Pics error: {e}")
