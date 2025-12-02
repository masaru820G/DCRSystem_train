#training gpuあり

from ultralytics import YOLO
def main():
    # Load a model
    model = YOLO("yolov8x.yaml")  # build a new model from scratch
    model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="0.yaml", epochs=100, device=0, patience=100)  # train the model imgsz=640
    #metrics = model.val()  # evaluate model performance on the validation set
    #results = model("")  # predict on an image
    #path = model.export(format="onnx")  # export the model to ONNX format


if __name__ == '__main__':
    #freeze_support() here if program needs to be frozen
    main() #execute this only when run directly, not when imported!
