#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: oedon
# GNU Radio version: 3.10.7.0

from packaging.version import Version as StrictVersion
from PyQt5 import Qt
from gnuradio import qtgui
import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import soapy
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
from wifi_phy_hier import wifi_phy_hier  # grc-generated hier_block
import foo
import ieee802_11
import sip



class IRS_user(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "IRS_user")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 1e6
        self.pdu_length = pdu_length = 50
        self.out_buf_size = out_buf_size = 96000
        self.multi_const = multi_const = 0.5
        self.interval = interval = 5
        self.freq = freq = 2600000000.0
        self.encoding = encoding = 2

        ##################################################
        # Blocks
        ##################################################

        # Create the options list
        self._samp_rate_options = [1000000.0, 5000000.0, 10000000.0, 20000000.0]
        # Create the labels list
        self._samp_rate_labels = ['1 MHz', '5 MHz', '10 MHz', '20 MHz']
        # Create the combo box
        self._samp_rate_tool_bar = Qt.QToolBar(self)
        self._samp_rate_tool_bar.addWidget(Qt.QLabel("'samp_rate'" + ": "))
        self._samp_rate_combo_box = Qt.QComboBox()
        self._samp_rate_tool_bar.addWidget(self._samp_rate_combo_box)
        for _label in self._samp_rate_labels: self._samp_rate_combo_box.addItem(_label)
        self._samp_rate_callback = lambda i: Qt.QMetaObject.invokeMethod(self._samp_rate_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._samp_rate_options.index(i)))
        self._samp_rate_callback(self.samp_rate)
        self._samp_rate_combo_box.currentIndexChanged.connect(
            lambda i: self.set_samp_rate(self._samp_rate_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._samp_rate_tool_bar)
        self._pdu_length_range = Range(10, 500, 1, 50, 200)
        self._pdu_length_win = RangeWidget(self._pdu_length_range, self.set_pdu_length, "'pdu_length'", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._pdu_length_win)
        self._multi_const_range = Range(0.2, 0.6, 0.05, 0.5, 200)
        self._multi_const_win = RangeWidget(self._multi_const_range, self.set_multi_const, "'multi_const'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._multi_const_win)
        self._interval_range = Range(2, 1500, 1, 5, 200)
        self._interval_win = RangeWidget(self._interval_range, self.set_interval, "'interval'", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._interval_win)
        # Create the options list
        self._freq_options = [4100000000.0, 2472000000.0, 3100000000.0, 2600000000.0, 2880000000.0, 3020000000.0, 3160000000.0, 3300000000.0, 3440000000.0, 3580000000.0, 3720000000.0, 3860000000.0]
        # Create the labels list
        self._freq_labels = ['  1 | 4100.0 | 11g', '  2 | 2472.0 | 11g', '  3 | 3100.0 | 11g', '  4 | 2600.0 | 11g', '  5 | 2880.0 | 11g', '  6 | 3020.0 | 11g', '  7 | 3160.0 | 11g', '  8 | 3300.0 | 11g', '  9 | 3440.0 | 11g', '  10 | 3580.0 | 11g', '  11 | 3720.0 | 11g', '  12 | 3860.0 | 11g']
        # Create the combo box
        self._freq_tool_bar = Qt.QToolBar(self)
        self._freq_tool_bar.addWidget(Qt.QLabel("'freq'" + ": "))
        self._freq_combo_box = Qt.QComboBox()
        self._freq_tool_bar.addWidget(self._freq_combo_box)
        for _label in self._freq_labels: self._freq_combo_box.addItem(_label)
        self._freq_callback = lambda i: Qt.QMetaObject.invokeMethod(self._freq_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._freq_options.index(i)))
        self._freq_callback(self.freq)
        self._freq_combo_box.currentIndexChanged.connect(
            lambda i: self.set_freq(self._freq_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._freq_tool_bar)
        # Create the options list
        self._encoding_options = [0, 1, 2, 3, 4, 5, 6, 7]
        # Create the labels list
        self._encoding_labels = ['BPSK 1/2', 'BPSK 3/4', 'QPSK 1/2', 'QPSK 3/4', '16QAM 1/2', '16QAM 3/4', '64QAM 2/3', '64QAM 3/4']
        # Create the combo box
        # Create the radio buttons
        self._encoding_group_box = Qt.QGroupBox("'encoding'" + ": ")
        self._encoding_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._encoding_button_group = variable_chooser_button_group()
        self._encoding_group_box.setLayout(self._encoding_box)
        for i, _label in enumerate(self._encoding_labels):
            radio_button = Qt.QRadioButton(_label)
            self._encoding_box.addWidget(radio_button)
            self._encoding_button_group.addButton(radio_button, i)
        self._encoding_callback = lambda i: Qt.QMetaObject.invokeMethod(self._encoding_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._encoding_options.index(i)))
        self._encoding_callback(self.encoding)
        self._encoding_button_group.buttonClicked[int].connect(
            lambda i: self.set_encoding(self._encoding_options[i]))
        self.top_layout.addWidget(self._encoding_group_box)
        self.wifi_phy_hier_0_0 = wifi_phy_hier(
            bandwidth=samp_rate,
            chan_est=ieee802_11.LS,
            encoding=ieee802_11.Encoding(encoding),
            frequency=freq,
            sensitivity=0.56,
        )
        self.soapy_hackrf_sink_0 = None
        dev = 'driver=hackrf'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_hackrf_sink_0 = soapy.sink(dev, "fc32", 1, '',
                                  stream_args, tune_args, settings)
        self.soapy_hackrf_sink_0.set_sample_rate(0, samp_rate)
        self.soapy_hackrf_sink_0.set_bandwidth(0, 0)
        self.soapy_hackrf_sink_0.set_frequency(0, freq)
        self.soapy_hackrf_sink_0.set_gain(0, 'AMP', False)
        self.soapy_hackrf_sink_0.set_gain(0, 'VGA', min(max(16, 0.0), 47.0))
        self.qtgui_sink_x_0 = qtgui.sink_c(
            1024, #fftsize
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "", #name
            True, #plotfreq
            True, #plotwaterfall
            True, #plottime
            True, #plotconst
            None # parent
        )
        self.qtgui_sink_x_0.set_update_time(1.0/10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.qwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(True)

        self.top_layout.addWidget(self._qtgui_sink_x_0_win)
        self.ieee802_11_mac_0 = ieee802_11.mac([0x23, 0x23, 0x23, 0x23, 0x23, 0x23], [0x42, 0x42, 0x42, 0x42, 0x42, 0x42], [0xff, 0xff, 0xff, 0xff, 0xff, 255])
        self.foo_packet_pad2_0 = foo.packet_pad2(False, False, 0.01, 100, 1000)
        self.foo_packet_pad2_0.set_min_output_buffer(out_buf_size)
        self.blocks_vector_source_x_0 = blocks.vector_source_c((0,), False, 1, [])
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(multi_const)
        self.blocks_multiply_const_vxx_0.set_min_output_buffer(100000)
        self.blocks_message_strobe_0_0 = blocks.message_strobe(pmt.intern("x".join('y' for i in range(pdu_length))), interval)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0_0, 'strobe'), (self.ieee802_11_mac_0, 'app in'))
        self.msg_connect((self.ieee802_11_mac_0, 'phy out'), (self.wifi_phy_hier_0_0, 'mac_in'))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.foo_packet_pad2_0, 0))
        self.connect((self.blocks_vector_source_x_0, 0), (self.wifi_phy_hier_0_0, 0))
        self.connect((self.foo_packet_pad2_0, 0), (self.qtgui_sink_x_0, 0))
        self.connect((self.foo_packet_pad2_0, 0), (self.soapy_hackrf_sink_0, 0))
        self.connect((self.wifi_phy_hier_0_0, 0), (self.blocks_multiply_const_vxx_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "IRS_user")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self._samp_rate_callback(self.samp_rate)
        self.qtgui_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.soapy_hackrf_sink_0.set_sample_rate(0, self.samp_rate)
        self.wifi_phy_hier_0_0.set_bandwidth(self.samp_rate)

    def get_pdu_length(self):
        return self.pdu_length

    def set_pdu_length(self, pdu_length):
        self.pdu_length = pdu_length
        self.blocks_message_strobe_0_0.set_msg(pmt.intern("x".join('y' for i in range(self.pdu_length))))

    def get_out_buf_size(self):
        return self.out_buf_size

    def set_out_buf_size(self, out_buf_size):
        self.out_buf_size = out_buf_size

    def get_multi_const(self):
        return self.multi_const

    def set_multi_const(self, multi_const):
        self.multi_const = multi_const
        self.blocks_multiply_const_vxx_0.set_k(self.multi_const)

    def get_interval(self):
        return self.interval

    def set_interval(self, interval):
        self.interval = interval
        self.blocks_message_strobe_0_0.set_period(self.interval)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self._freq_callback(self.freq)
        self.soapy_hackrf_sink_0.set_frequency(0, self.freq)
        self.wifi_phy_hier_0_0.set_frequency(self.freq)

    def get_encoding(self):
        return self.encoding

    def set_encoding(self, encoding):
        self.encoding = encoding
        self._encoding_callback(self.encoding)
        self.wifi_phy_hier_0_0.set_encoding(ieee802_11.Encoding(self.encoding))




def main(top_block_cls=IRS_user, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
