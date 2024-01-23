# Image Captioning Streamlit App

This Streamlit app uses a pre-trained Vision Encoder-Decoder model based on ViT-GPT2 for image captioning. Users can upload an image, and the app will generate a caption for the image.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your-username/your-image-captioning-app.git
    cd image_captioning
    ```

2. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```
    streamlit run app.py
    ```

4. Open your browser and go to [http://localhost:8501](http://localhost:8501) to access the app.

## Usage

1. Upload an image using the file uploader.
2. Click the "Generate Caption" button.
3. View the generated caption for the uploaded image.

## Dependencies

- torch
- transformers
- streamlit
- Pillow

## Credits

- The image captioning model is based on the Hugging Face Transformers library.
- Streamlit is used for building the user interface.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
