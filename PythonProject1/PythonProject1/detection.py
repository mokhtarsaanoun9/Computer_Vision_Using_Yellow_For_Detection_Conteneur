if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import torch
    from ultralytics import YOLO

    model = YOLO("yolov8n.yaml")  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train(
        data='C:/Users/wijde/PycharmProjects/PythonProject1/detection/data.yaml',
        epochs=40,
        imgsz=640,
        device=device
    )


