## Integrtar our code with OpenAi Api
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlist framework

st.title('Celeberity Search Results')
input_text=st.text_input("Search the topic you want")

# Promt Template

fisrt_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)



## OpenAi LLMs
llm=OpenAI(temperature=0.8)

chain=LLMChain(llm=llm,prompt=fisrt_input_prompt,verbose=True,output_key='person')

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="When was {person} born"
)

chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob')

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened in that year {dob} in the world",
)

chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='events')

chains=SequentialChain(
    chains=[chain,chain2,chain3],input_variables=['name'], output_variables=['person','dob','events'],verbose=True)


if input_text:
    st.write(chains({'name':input_text}))


