## Integrtar our code with OpenAi Api
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title('Customer Service Center')
user_question=st.text_input("Ask your questions here.....")

 
loader = TextLoader("custom-data.txt")
#loader = DirectoryLoader('<Directory>')
#loader.load()

index = VectorstoreIndexCreator().from_loaders([loader])
output_text=index.query(user_question, ChatOpenAI())
st.write(output_text)
