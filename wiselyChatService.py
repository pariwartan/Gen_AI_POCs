import streamlit as st
import os
from dotenv import load_dotenv
from constants import openai_key
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
#from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, page_bg_img
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from PIL import Image
from langchain.prompts import PromptTemplate


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
    embeddings = OpenAIEmbeddings() # using TikToken for tokenization (Byte Pair Encoding), it can count toekn count which will helpful for cost estimation
    #vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    #vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings, persist_directory="vector_db")
    #vectorstore.persist()
    #Chroma.from_documents(docs, embeddings)
    vectorstore =Chroma(persist_directory="vector_db", embedding_function= embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.8) 
    #llm = OpenAI(temperature=0.8, top_p= 0.6 )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    prompt_template="""You are a helpful assitant.If the context is not relevent then answer using your own knowledge base
                       {context}
                       Question: {question}
                    """
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': PromptTemplate(template=prompt_template, input_variables=["context","question"])}
    )
    return conversation_chain


def handle_userinput(user_question):
    with st.spinner("Processing request..."):
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
    logo_image = Image.open("images/logo-wisely-white.png")
    st.set_page_config(page_title="Wisely Customer Service Center",
                       page_icon="images/logo-wisely-color.png")
    st.image(logo_image, width=100)
    st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "data_loaded" not in st.session_state:

        # get Wisely Data
        raw_text = get_wisely_public_data()

        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # create vector store
        vectorstore = get_vectorstore(text_chunks)

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(
            vectorstore)
        
        st.session_state.data_loaded = "data_loaded"
        

    st.header("Wisely Customer Service Center :phone:")
    st.markdown(page_bg_img, unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about Wisely:")
 
    if user_question:
        handle_userinput(user_question)



if __name__ == '__main__':
    main()
