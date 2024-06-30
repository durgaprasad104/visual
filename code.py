import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, ViltForQuestionAnswering
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import av

# Load VQA model, processor, and tokenizer from Hugging Face
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
tokenizer = AutoTokenizer.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Streamlit app
st.title("Visual Question Answering")

st.write("Upload an image or capture a live image and ask a question about it.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Placeholder for image
image = None

# Class to process the video stream
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.image = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.image = Image.fromarray(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit WebRTC streamer
ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)

# Display captured image from webcam
if ctx.video_transformer:
    image = ctx.video_transformer.image
    if image is not None:
        st.image(image, caption='Captured Image', use_column_width=True)

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Question input
question = st.text_input("Ask a question about the image")

if image is not None and question:
    # Prepare inputs for the VQA model with padding
    inputs = processor(images=image, text=question, return_tensors="pt", padding=True, truncation=True)

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_answer_idx = torch.argmax(logits, dim=1).item()

    # Use id2label to get the human-readable answer
    answer = model.config.id2label.get(predicted_answer_idx, f"[index {predicted_answer_idx}]")

    st.write("Answer:", answer)
