import os
import time
import aiohttp
import requests
import asyncio
import streamlit as st
from typing import Any
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_text_from_image(url: str) -> str:
    image_to_text: Any = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    generated_text: str = image_to_text(url)[0]["generated_text"]

    print(f"IMAGE INPUT: {url}")
    print(f"GENERATED TEXT OUTPUT: {generated_text}")
    return generated_text


def generate_story_from_text(scenario: str) -> str:
    prompt_template: str = f"""
    You are a talented story teller who can create a story from a simple narrative./
    Create a story using the following scenario; the story should have be maximum 100 words long;
    
    CONTEXT: {scenario}
    STORY:
    """

    prompt: PromptTemplate = PromptTemplate(template=prompt_template, input_variables=["scenario"])
    llm: Any = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    story_llm: Any = LLMChain(llm=llm, prompt=prompt, verbose=True)
    generated_story: str = story_llm.predict(scenario=scenario)

    print(f"TEXT INPUT: {scenario}")
    print(f"GENERATED STORY OUTPUT: {generated_story}")
    return generated_story


async def generate_speech_from_text(message: str) -> None:
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads = {
        "inputs": message,
        "options": {
            "use_cache": False
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payloads) as response:
            if response.status == 200:
                content = await response.read()
                with open("generated_audio.wav", "wb") as file:
                    file.write(content)
            else:
                error_message = await response.json()
                st.error(f"Failed to generate speech: {error_message}")


async def main_async(uploaded_file: Any):
    progress_text = "Please wait! Model is preparing the story based on uploaded image..."
    my_bar = st.progress(0, text=progress_text)
    
    bytes_data: Any = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Update progress bar during each major step
    my_bar.progress(25, text="Generating text from image...")
    scenario: str = await asyncio.to_thread(generate_text_from_image, uploaded_file.name)
    
    my_bar.progress(50, text="Generating story from text...")
    story: str = await asyncio.to_thread(generate_story_from_text, scenario)
    
    my_bar.progress(75, text="Generating speech from story...")
    await generate_speech_from_text(story)
    
    my_bar.progress(100, text="Completed")

    with st.expander("Generated Image scenario"):
        st.write(scenario)
    with st.expander("Generated short story"):
        st.write(story)

    st.audio("generated_audio.wav")


def main() -> None:
    st.set_page_config(page_title="STORY TELLER")
    st.header("Story Teller")
    uploaded_file: Any = st.file_uploader("Please choose a file to upload", type=["jpg", "jpeg", "png", "gif", "bmp"])

    if uploaded_file is not None:
        asyncio.run(main_async(uploaded_file))


if __name__ == "__main__":
    main()
