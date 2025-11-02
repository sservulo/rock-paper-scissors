from ultralytics import YOLO

LOCAL_DATASET_PATH = "/home/samuel/rock-paper-scissor-dataset/data.yaml"


def rps():
    model = YOLO("yolo11n.pt")
    results = model.train(data=LOCAL_DATASET_PATH, epochs=3)
    results = model.val()  # evaluate model performance
    model.export()


if __name__ == "__main__":
    rps()
