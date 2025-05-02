## Table of Contents

* [ Project Structure](#-project-structure)

  * [ Project Index](#-project-index)
* [ Getting Started](#-getting-started)

  * [ Prerequisites](#-prerequisites)
  * [ Installation](#-installation)
  * [ OpenPose Setup](#-openpose-setup)
  * [ HRNet Setup](#-hrnet-setup)
  * [ Usage](#-usage)
---
## Project Structure

```sh
└── aspose/
    ├── analysis
    │   ├── config.py
    │   ├── main.py
    │   ├── maths_helpers.py
    │   ├── pose_backends
    │   └── requirements.txt
    ├── backend
    │   ├── backend
    │   ├── core
    │   └── manage.py
    └── frontend
        ├── .gitignore
        ├── README.md
        ├── eslint.config.js
        ├── index.html
        ├── package-lock.json
        ├── package.json
        ├── src
        └── vite.config.js
```

---

## Getting Started

### Prerequisites

Before getting started with aspose, ensure your runtime environment meets the following requirements:

* **Programming Language:** Python
* **Package Manager:** Pip, Npm

### Installation

Install ASPose using the following steps:

```sh
❯ git clone https://github.com/ch020/aspose
❯ cd aspose
❯ pip install -r analysis/requirements.txt
❯ cd frontend
❯ npm install
```

### OpenPose Setup

1. Download the Windows build of OpenPose from the official repository: [OpenPose Releases](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases).
2. Extract the contents and place `OpenPoseDemo.exe` and its associated files into:

```
analysis/openpose/
```

3. Test that `OpenPoseDemo.exe` works independently before calling it from within ASPose.

### HRNet Setup

1. Clone the HRNet repository into the `analysis/` directory:

```sh
❯ cd analysis
❯ git clone https://github.com/HRNet/HRNet-Human-Pose-Estimation-Pytorch hrnet
```

2. Download the pretrained model [`pose_hrnet_w32_256x192.pth`](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC) and place it inside:

```
analysis/hrnet/
```

3. Modify `analysis/config.py` to specify which pose estimation backend (e.g. "openpose" or "hrnet") should be used and point to any required paths.

### Usage

To run the analysis pipeline:

```sh
❯ python analysis/main.py
```

To run the test suite:

```sh
❯ pytest
```
