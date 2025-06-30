# SDR-based Image Transmission Project using GNU Radio

This project demonstrates image transmission and reception using Software Defined Radios (SDRs) and GNU Radio, leveraging the IEEE 802.11 PHY layer implementation. It includes functionalities for both standard single-image transmission and a specialized mode for transmitting fused images, intended as supplementary material related to the JSCE research paper.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Project](#running-the-project)
  - [1. Connect Hardware and Start GNU Radio Flowgraphs](#1-connect-hardware-and-start-gnu-radio-flowgraphs)
  - [2. Start Control Scripts](#2-start-control-scripts)
    - [Mode 1: Fused Image Transmission (JSCE Supplement)](#mode-1-fused-image-transmission-jsce-supplement)
    - [Mode 2: Traditional Single Image Transmission](#mode-2-traditional-single-image-transmission)
- [How it Works (Briefly)](#how-it-works-briefly)
- [Connection to Research](#connection-to-research)
- [License](#license)

## Prerequisites

*   **GNU Radio:** An installed and functional version of GNU Radio (preferably 3.8 or later).
*   **SDR Hardware:** At least one SDR device (USRP, HackRF, etc.). For full duplex operation (simultaneous TX/RX), two devices or one full-duplex device are needed.
*   **Python 3:** Installed Python 3 environment.
*   **Required Python Libraries:** Install necessary libraries using pip. A `requirements.txt` file is recommended, but likely includes: `flask`, `socket`, `numpy`, `Pillow` (or `opencv`). You can install them via:
    ```bash
    pip install -r requirements.txt
    ```
    *(Assuming you create a `requirements.txt` file listing the dependencies)*
*   **GNU Radio IEEE 802.11/WiFi Module:** This project relies on this module. If it's not installed system-wide, you will need to import it as a hierarchical block as described in the Setup section.

## Setup

If you do not have the IEEE 802.11/WiFi module installed system-wide for GNU Radio, you need to make it available as a hierarchical block:

1.  Open GNU Radio Companion (GRC).
2.  Go to `File` -> `Import`.
3.  Navigate to the project directory `gnu_radio` and select the `wifi_phy_hier.grc` file.
4.  Importing this file makes the hierarchical block `wifi_phy_hier` available in your GRC block tree. This block is used within the `IRS_AP.grc` and `IRS_user.grc` flowgraphs.

## Running the Project

Follow these steps to get the image transmission system running:

### 1. Connect Hardware and Start GNU Radio Flowgraphs

1.  Connect your SDR hardware (USRP, HackRF, etc.) to your computer(s).
2.  Open the GNU Radio Companion (`gnuradio-companion`).
3.  Open the flowgraphs:
    *   `IRS_AP.grc` (This typically acts as the **Receiver / Access Point**)
    *   `IRS_user.grc` (This typically acts as the **Transmitter / User**)
4.  **Hardware Configuration:** Inside each flowgraph, ensure the correct SDR Source/Sink blocks are configured for your connected hardware (e.g., `UHD USRP Source/Sink` for USRP, `OsmoSDR Source/Sink` for HackRF). The default setup assumes:
    *   HackRF connected to `IRS_user.grc` (Transmitter)
    *   USRP connected to `IRS_AP.grc` (Receiver)
    *   **Note:** You can adjust the source/sink blocks in the `.grc` files to match your available hardware and desired TX/RX roles.
5.  Execute (run) both `IRS_AP.grc` and `IRS_user.grc` flowgraphs in GRC. They should start and wait for data to be sent via the Socket PDU blocks.

### 2. Start Control Scripts

Now, open new terminal windows for the Python control scripts. These scripts interact with the running GNU Radio flowgraphs via UDP Socket PDU blocks to send and receive image data.

You can choose between two transmission modes:

#### Mode 1: Fused Image Transmission (JSCE Supplement)

This mode is specifically designed for transmitting fused images, used as supplementary material or a demonstration related to the JSCE research paper.

1.  **Start the Send Window (Flask Server):**
    ```bash
    python upload_featuremap_udp.py
    ```
    This script starts a Flask web server. Open the URL provided by the Flask app in your web browser (usually `http://127.0.0.1:5000/`). You will use this interface to select and upload the two images you want to fuse and transmit.
2.  **Start the Receive Window:**
    ```bash
    python download_featuremap_udp.py
    ```
    This script listens for incoming data from the receiver GNU Radio flowgraph and will display the received image.

**Socket PDU Port:** The `upload_featuremap_udp.py` script sends image data *to* the `IRS_user.grc` flowgraph via a Socket PDU (UDP) block. Ensure the IP address and Port configured in the `upload_featuremap_udp.py` script matches the configuration of the Socket PDU block in your `IRS_user.grc` flowgraph (which should be set to 'UDP Client' mode, sending to the address/port the script is listening on, typically localhost). *Conversely, the `download_featuremap_udp.py` script listens for data from `IRS_AP.grc`, so the Socket PDU in `IRS_AP.grc` should be 'UDP Client' sending to the address/port `download_featuremap_udp.py` is listening on.*

#### Mode 2: Traditional Single Image Transmission

This mode is for standard transmission of a single image.

1.  **Start the Send Window:**
    ```bash
    python upload_image_udp.py
    ```
    This script will likely provide an interface (either command-line or GUI) to select a single image for transmission.
2.  **Start the Receive Window:**
    ```bash
    python download_image_udp.py
    ```
    This script listens for incoming data and will display the received single image.

**Note:** In this traditional mode, only one image is transmitted at a time. The communication with the GNU Radio flowgraphs (`IRS_user.grc` and `IRS_AP.grc`) still happens via the Socket PDU mechanism, similar to Mode 1, but using the `upload_image_udp.py` and `download_image_udp.py` scripts. Ensure the Socket PDU ports match between the scripts and the `.grc` files.

## How it Works (Briefly)

The system operates as follows:

1.  The Python upload script (`upload_featuremap_udp.py` or `upload_image_udp.py`) reads the image file(s).
2.  It processes the image data and sends it as UDP packets to the Socket PDU block configured in the `IRS_user.grc` flowgraph.
3.  `IRS_user.grc` receives the UDP packets, processes the data through the IEEE 802.11 PHY chain, and transmits the resulting waveform via the connected SDR (e.g., HackRF).
4.  `IRS_AP.grc` receives the waveform via its connected SDR (e.g., USRP), processes it through the IEEE 802.11 PHY chain, and outputs the received data via its Socket PDU block.
5.  The Python download script (`download_featuremap_udp.py` or `download_image_udp.py`) listens for UDP packets from the `IRS_AP.grc` flowgraph's Socket PDU block.
6.  It receives the data, reconstructs the image, and displays it.

## Connection to Research

The "Fused Image Transmission" mode (`upload_featuremap_udp.py` / `download_featuremap_udp.py`) is developed to complement or provide a practical testbed for concepts discussed in the JSCE research paper:
```
@misc{wang2025learningjointsourcechannelencoding,
      title={Learning Joint Source-Channel Encoding in IRS-assisted Multi-User Semantic Communications}, 
      author={Haidong Wang and Songhan Zhao and Lanhua Li and Bo Gu and Jing Xu and Shimin Gong and Jiawen Kang},
      year={2025},
      eprint={2504.07498},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2504.07498}, 
}
```


It specifically demonstrates the capability to realize multiplexing using semantic transmission.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
