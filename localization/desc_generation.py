import base64
from openai import OpenAI, RateLimitError, AuthenticationError
from typing import Tuple
from tqdm import tqdm
import pandas as pd
import glob
import os

def get_image_data_url(image_file: str, image_format: str) -> str:
    """
    Helper function to converts an image file to a data URL string.
    Args:
        image_file (str): The path to the image file.
        image_format (str): The format of the image file.
    Returns:
        str: The data URL of the image.
    """
    try:
        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Could not read '{image_file}'.")
        return None
    return f"data:image/{image_format};base64,{image_data}"


def get_image_description(image_file: str, api_key: str, base_url: str = None) -> Tuple[str, Exception]:
    """
    Helper function to get image description from OpenAI API.
    Args:
        image_file (str): The path to the image file.
        client (OpenAI): The OpenAI client instance.
    Returns:
        str: The description of the image.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None
    )
    image_url = get_image_data_url(image_file, "jpg")

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that describes images in detail.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Is there anything unrealistic in this image? yes or no or somewhat,if yes or somewhat explain in maximum 30 words, please ensure to explain what looks unreal like if face is distorted , or transition between objects is not smooth, or any other unrealistic thing you see in the image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ],
                },
            ],
            model="gpt-4.1",
        )

    except RateLimitError as e:
        print(f"Rate limit exceeded for API key. Try again later.")
        return "", e
    
    except AuthenticationError as e:
        print(f"Authentication error with API key. Use a different API key.")
        return "", e
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", e
    return response.choices[0].message.content, None
