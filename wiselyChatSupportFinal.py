import streamlit as st
import os
from dotenv import load_dotenv
from constants import openai_key
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
#from langchain.llms import OpenAI
#from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
#from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, page_bg_img
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader


os.environ["OPENAI_API_KEY"]=openai_key

def get_wisely_public_data():
    text = ""
    data = TextLoader("custom-data.txt")
    #loader = DirectoryLoader('<Directory>')
    #loader.load()
    docs = data.load()
    for doc in docs:
        text += doc.page_content

    return text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    #Chroma.from_documents(docs, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.8) 
    #llm = OpenAI(temperature=0.8, top_p= 0.6 )

    memory = ConversationBufferWindowMemory(
        memory_key='chat_history', return_messages=True, K=3) # K is a window size, Its going to store last three conversations 
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv("constants.py")
    st.set_page_config(page_title="Wisely Customer Service Center",
                       page_icon="images/logo-wisely-color.png")
    st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = ["Welcome to Wisely Support, How may i assist yoy today?"]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Wisely Customer Service Center :images/logo-wisely-color.png:")
    st.markdown(page_bg_img, unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about Wisely:")
    if user_question:
        handle_userinput(user_question)

    # get Wisely Data
    raw_text = get_wisely_public_data()

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        vectorstore)


if __name__ == '__main__':
    main()
