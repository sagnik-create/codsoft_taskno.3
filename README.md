# codsoft_taskno.3
# Image Caption Generator

## Overview
This project is an Image Caption Generator built using Python and Streamlit. The application leverages a pre-trained VisionEncoderDecoder model to generate captions for uploaded images.

## Features
- **Upload Image**: User can upload an image file in JPG, JPEG, PNG, or WEBP format.
- **Caption Generation**: The application generates and displays a caption for the uploaded image using a pre-trained model.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required dependencies:
    ```bash
    pip install streamlit Pillow transformers torch
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to interact with the Image Caption Generator.

## Code Explanation
### Import Libraries
The necessary libraries are imported, including Streamlit, PIL for image processing, and Transformers for the captioning model:
```python
import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
```

### Load Pre-trained Model
The pre-trained VisionEncoderDecoder model, feature extractor, and tokenizer are loaded:
```python
captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
```

### Streamlit App
A simple Streamlit app is created to interact with the user:
```python
st.title("Image Caption Generator")
st.write("Upload an image to generate a caption.")
```

### Image Upload
The user can upload an image file:
```python
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
```

### Caption Generation
If an image is uploaded, it is displayed, and a caption is generated and shown to the user:
```python
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values

    caption_ids = captioning_model.generate(pixel_values, max_length=16, num_beams=4, early_stopping=True)
    caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)

    st.write("### Generated Caption:")
    st.write(caption)
```

## Contributing
Feel free to fork this project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [Pillow (PIL)](https://python-pillow.org/)
- [Transformers](https://huggingface.co/transformers/)
- [Torch](https://pytorch.org/)
