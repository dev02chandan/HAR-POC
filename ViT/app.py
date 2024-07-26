import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import tempfile
import torch
# import ffmpeg   

# Initialize the pipeline with the pre-trained model
pipe = pipeline("image-classification", "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224", device=0 if torch.cuda.is_available() else -1)

def process_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return pipe(image)

def annotate_frame(frame, result):
    top_score = result[0]['score']
    if top_score > 0.8:
        num_labels = 1
    elif 0.5 < top_score <= 0.8:
        num_labels = 2
    else:
        num_labels = 3
    
    text = ', '.join([f"{result[i]['label']}: {result[i]['score']:.2f}" for i in range(min(num_labels, len(result)))])

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 30
    cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_output = 'temp_output.mp4'
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            result = process_frame(frame)
            annotated_frame = annotate_frame(frame, result)
            out.write(annotated_frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Convert to H.264 format
    h264_output = 'output_video.mp4'
    ffmpeg.input(temp_output).output(h264_output, vcodec='libx264').run(overwrite_output=True)
    return h264_output

def main():
    st.title("Human Activity Recognition")
    st.write("Upload an image, a video, or use your webcam to detect human activities.")

    # Image Upload
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        result = process_frame(np.array(image))
        annotated_image = annotate_frame(np.array(image), result)
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
        st.write(result)

    # Video Upload
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        st.write("Processing video...")
        output_video_path = process_video(video_path)
        st.video(output_video_path)

    # Webcam
    use_webcam = st.checkbox('Use webcam')
    if use_webcam:
        st.write("Opening webcam...")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                result = process_frame(frame)
                annotated_frame = annotate_frame(frame, result)
                stframe.image(annotated_frame, channels="BGR")
            else:
                break

        cap.release()

if __name__ == '__main__':
    main()
