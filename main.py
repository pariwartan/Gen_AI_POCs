## Integrtar our code with OpenAi Api
import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlist framework

st.title('LangChain Demo with OpenAI Api')
input_text=st.text_input("Search the topic you want")

## OpenAi LLMs
llm=OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))


