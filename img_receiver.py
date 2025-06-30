import pdb

import pmt
import socket
import numpy as np
import pickle
from gnuradio import gr
from PIL import Image


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    def __init__(self,
                 image_hight: int = 300,
                 image_width: int = 300):  # only default arguments here
        gr.basic_block.__init__(
            self,
            name='Extract Pics',  # will be shown in GRC
            in_sig=None,
            out_sig=None
        )
        self.message_port_register_in(pmt.intern('MAC'))
        self.set_msg_handler(pmt.intern('MAC'), self.handle_msg)
        self.skt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.H = image_hight
        self.W = image_width

    def handle_msg(self, msg):
        # This is where you handle the message. You can access the data using pmt.to_python(msg).
        # global patch_list
        try:
            data = pmt.to_python(msg)[-1]
            # print(data)
            # pdb.set_trace()
            target = data[24:]  # loaded data
            pics = target[4:]
            
            img = bytes(pics)
            # img = pickle.dumps(pics)
            self.skt.sendto(img, ('localhost', 10010))
        except UnicodeDecodeError as e:
            print(f"An error occurred: {e}")
            # pass
        # print(f'msg received, current msg list length {len(patch_list)}')

