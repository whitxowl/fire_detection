import cv2
import os
import PIL.Image as Image
import gradio as gr
import numpy as np
from ultralytics import YOLO

model = YOLO("best.onnx")

def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

image_directory = "/home/user/app/image"
video_directory = "/home/user/app/video"

image_iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold")
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Fire Detection using YOLOv8n on Gradio",
    description="Upload images for inference. The Ultralytics YOLOv8n trained model is used for this.",
    examples=[
        [os.path.join(image_directory, "fire_image_1.jpg"), 0.25, 0.45],
        [os.path.join(image_directory, "fire_image_3.jpg"), 0.25, 0.45],
        
    ]
)

def pil_to_cv2(pil_image):
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return open_cv_image

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pil_img = Image.fromarray(frame[..., ::-1])
        result = model.predict(source=pil_img)
        for r in result:
            im_array = r.plot()
            processed_frame = Image.fromarray(im_array[..., ::-1])
        yield processed_frame
    cap.release()
    
video_iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Video", interactive=True)
    ],
    outputs=gr.Image(type="pil",label="Result"),
    title="Fire Detection using YOLOv8n on Gradio",
    description="Upload video for inference. The Ultralytics YOLOv8n trained model is used for inference.",
    examples=[
        [os.path.join(video_directory, "video_fire_1.mp4")],
        [os.path.join(video_directory, "video_fire_2.mp4")],
    ]
)


demo = gr.TabbedInterface([image_iface, video_iface], ["Image Inference", "Video Inference"])

if __name__ == '__main__':
    demo.launch()