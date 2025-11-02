# Rock Paper Scissors

## Requirements

### Data

Dataset: https://universe.roboflow.com/group-project-2-yolov5/rock-paper-scissor-usgmy

Frameworks: YOLO/Pytorch (ultralytics)

### UV

For environment setup.

In the terminal, install uv:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

And update it:
```
uv self update
```

## Running

To train:

```
uv run train.py
```

To run live detection:

```
uv run live.py
```