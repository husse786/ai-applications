import base64
import json
import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Load models
vit_classifier = pipeline("image-classification", model="kuhs/vit-base-oxford-iiit-pets")
clip_detector = pipeline(model="openai/clip-vit-large-patch14", task="zero-shot-image-classification")

labels_oxford_pets = [
    'Siamese', 'Birman', 'shiba inu', 'staffordshire bull terrier', 'basset hound', 'Bombay', 'japanese chin',
    'chihuahua', 'german shorthaired', 'pomeranian', 'beagle', 'english cocker spaniel', 'american pit bull terrier',
    'Ragdoll', 'Persian', 'Egyptian Mau', 'miniature pinscher', 'Sphynx', 'Maine Coon', 'keeshond', 'yorkshire terrier',
    'havanese', 'leonberger', 'wheaten terrier', 'american bulldog', 'english setter', 'boxer', 'newfoundland', 'Bengal',
    'samoyed', 'British Shorthair', 'great pyrenees', 'Abyssinian', 'pug', 'saint bernard', 'Russian Blue', 'scottish terrier'
]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def classify_with_openai(image_path):
    if openai_client is None:
        return {
            "error": "Missing OPENAI_API_KEY. Add it to your environment or .env file to enable OpenAI classification."
        }

    prompt = (
        "Classify the pet in this image. Choose the best matching label from this list: "
        f"{', '.join(labels_oxford_pets)}. "
        "Return valid JSON with exactly these keys: "
        "label, confidence, reasoning. "
        "The confidence must be a number between 0 and 1."
    )

    base64_image = encode_image(image_path)
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    try:
        parsed_response = json.loads(response.output_text)
    except json.JSONDecodeError:
        parsed_response = {
            "raw_response": response.output_text,
            "warning": "OpenAI response was not valid JSON.",
        }

    return parsed_response


def classify_pet(image):
    vit_results = vit_classifier(image)
    vit_output = {result['label']: result['score'] for result in vit_results}
    
    clip_results = clip_detector(image, candidate_labels=labels_oxford_pets)
    clip_output = {result['label']: result['score'] for result in clip_results}
    
    openai_output = classify_with_openai(image)

    return {
        "ViT Classification": vit_output,
        "CLIP Zero-Shot Classification": clip_output,
        "OpenAI Vision Classification": openai_output,
    }

example_images = [
    ["example_images/dog1.jpeg"],
    ["example_images/dog2.jpeg"],
    ["example_images/leonberger.jpg"],
    ["example_images/snow_leopard.jpeg"],
    ["example_images/cat.jpg"]
]

iface = gr.Interface(
    fn=classify_pet,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),
    title="Pet Classification Comparison",
    description="Upload an image of a pet, and compare results from a trained ViT model, a zero-shot CLIP model, and an OpenAI vision model.",
    examples=example_images
)

iface.launch()