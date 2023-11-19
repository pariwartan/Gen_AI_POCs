import os
import pymysql
from google.cloud import vision
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from constants import openai_key
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import PyPDF2
import spacy
import fitz
from flask import Flask, render_template, request,session, redirect, url_for
import autogen
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
import json
from langchain.schema.messages import HumanMessage, AIMessage
import base64


# Set your Google Cloud credentials environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'vision_api.json'

config_list_4v = [
    {
        "model": "gpt-4-vision-preview",
        "api_key": "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"
    }
]


os.environ["OPENAI_API_KEY"]=openai_key

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.secret_key = "super secret key"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def start():
    if request.method == "POST":
        return render_template('Vai_start.html')
    else:   
        return render_template('Vai_start.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        return render_template('Vai_start.html')
    else:   
        return render_template('Vai_start.html')

@app.route('/upload_SO', methods=['GET', 'POST'])
def upload_SO():
    if request.method == 'POST':
        #prompt = request.form['prompt']

        customer = request.form.get("customer")
        session['customer'] = customer        

        if 'file' not in request.files:
            return redirect(request.url)
    
        file = request.files['file']


        if file.filename == '':
            return redirect(request.url)
    
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        file.save(filename)

        SO_text = extract_text_from_pdf(filename)

        session['SO_text'] = SO_text

        if customer.replace(" ", "") in SO_text.replace(" ", ""):
            print(SO_text)
            return render_template('Vai_KYC.html', SO_text=SO_text, customer=customer , show_hidden=False)
        else:
             return render_template('Vai_SO_failed.html', SO_text=SO_text , show_hidden=False)           
    else:
        return render_template('Vai_start.html')

@app.route('/upload_KYC', methods=['GET', 'POST'])
def upload_KYC():
    if request.method == 'POST':
        #prompt = request.form['prompt']

        if 'file' not in request.files:
            return redirect(request.url)
    
        file = request.files['file']


        if file.filename == '':
            return redirect(request.url)
    
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        file.save(filename)
        
        kyc_data = {}

        #KYC_Text = extract_text_from_image_GPT4v(filename) ## GPT4v with Autogen
        KYC_Text = extract_text_from_image_langchain_GPT4v(filename)  ## GPT4v with Langchain
        #KYC_Text = extract_text_from_image(filename) ## Google vision api with Langchain
        kyc_data = json.loads(KYC_Text)
        print(kyc_data)
        print("kyc_data----------------------------------")
        print(kyc_data)

        customer = kyc_data['Name']

        print(customer)

        SO_text = session['SO_text']

        if customer.replace(" ", "") in SO_text.replace(" ", ""):
            save_customer(KYC_Text, SO_text)
            return render_template('Vai_KYC_valid.html', kyc_data=kyc_data, customer=customer , show_hidden=False)
        else:
             return render_template('Vai_KYC_failed.html', kyc_data=kyc_data, customer=customer , show_hidden=False)           
    else:
        return render_template('index_final.html')           


def save_customer(KYC_Text, SO_text):          
        db_user = "root"
        db_password = "Admin"
        db_host = "localhost"
        db_name = "gen_ai"
        sql_db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        # Setup the database chain
        llm = ChatOpenAI(temperature=0.8, top_p= 0.6 )
        db_chain = SQLDatabaseChain(llm=llm, database=sql_db, verbose=True)

        # Create db chain
        select_QUERY = """
        Given an input question, first create a syntactically correct mysql query to run,
        then look at the results of the query and return the answer.
        Use the following format:

        Question: Question here
        SQLQuery: SQL Query to run
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        {question}
        """ 

        # Create db chain
        insert_QUERY = """
        insert a record in customer db,
        take input from these two texts.
        text1: {SO_text}
        text2: {KYC_Text}
        """ 

        question = insert_QUERY.format(SO_text=SO_text,KYC_Text=KYC_Text)
        print(db_chain.run(question))
        return True

def extract_text_from_image_langchain_GPT4v(img_path):
        image = encode_image(img_path)
        chain=ChatOpenAI(model="gpt-4-vision-preview",max_tokens=1024)
        msg = chain.invoke(
            [   AIMessage(
                content="You are a useful bot that is especially good at OCR from images"
            ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "what are the texts written in the image, Do not add any other information in the response."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            },
                        },
                    ]
                )
            ]
        )
        llm = ChatOpenAI(temperature=0.1)
        #prompt = PromptTemplate("""identify name address etc from given text and put it in key value pair. text : {extracted_text}""")
        prompt=PromptTemplate(
            input_variables=['extracted_text'],
            template="identify name,address,dob and other content from given text and put it in key value pair. text : {extracted_text}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        extracted_text = msg.content
        print(extracted_text)
        print("extracted_text printed------------------------------------")
        output = chain.run(extracted_text)
        print(output)
        print("output printed------------------------------------")
        return output

def extract_text_from_image_autogen_GPT4v(img_path):

    image_agent = MultimodalConversableAgent(
            name="image-explainer",
            max_consecutive_auto_reply=10,
            llm_config={"config_list": config_list_4v, "temperature": 0.5, "max_tokens": 300}
    )

    user_proxy = autogen.UserProxyAgent(
            name="User_proxy",
            system_message="A human admin.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0
    )

    message="""Please read the text in this image and return the information, Please do not add any other information in the response
            <img {img_path}>.
            """
    message= message.format(img_path=img_path)

    # Ask the question with an image  
    user_proxy.system_message()
    extracted_text = user_proxy.initiate_chat(image_agent, 
                            message=message)
    print(extracted_text)
    llm = ChatOpenAI(temperature=0.1)
    #prompt = PromptTemplate("""identify name address etc from given text and put it in key value pair. text : {extracted_text}""")
    prompt=PromptTemplate(
        input_variables=['extracted_text'],
        template="identify name,address,dob and other content from given text and put it in key value pair. text : {extracted_text}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(extracted_text)
    print(output)
    return output
  

def extract_text_from_image(image_path):
    # Create a Vision API client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    # Create an Image object
    image = vision.Image(content=content)

    # Analyze the image using the Safe Search Detection feature to check for explicit content
    safe_search_response = client.safe_search_detection(image=image)
    
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = (
        "UNKNOWN",
        "VERY_UNLIKELY",
        "UNLIKELY",
        "POSSIBLE",
        "LIKELY",
        "VERY_LIKELY",
    )
    # Check for tampering based on the Safe Search results
    safe_search_annotation = safe_search_response.safe_search_annotation
    print(likelihood_name[safe_search_annotation.spoof])
    if likelihood_name[safe_search_annotation.spoof] == "LIKELY":
        print("Document may be tampered (Likely Spoofing).")
    elif likelihood_name[safe_search_annotation.spoof] == "VERY_LIKELY":
        print("Document is likely tampered (Very Likely Spoofing).")
    else:
        print("Document appears to be intact.")

    # Perform Optical Character Recognition (OCR) on the image
    response = client.text_detection(image=image)

    # Extract text from the response
    texts = response.text_annotations
    if texts:
        # The first element contains the detected text
        extracted_text = texts[0].description
        print(extracted_text)
        llm = ChatOpenAI(temperature=0.1)
        #prompt = PromptTemplate("""identify name address etc from given text and put it in key value pair. text : {extracted_text}""")
        prompt=PromptTemplate(
            input_variables=['extracted_text'],
            template="identify name,address,dob and other content from given text and put it in key value pair. text : {extracted_text}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.run(extracted_text)
        print(output)
        return output
    else:
        return "No text found in the image."

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    print(pdf_path)
    with open(pdf_path, 'rb') as file:
        reader = fitz.open(file)
        text = ''
        for page in reader:
            page_text = page.get_text()
            text += page_text
    return text

# Function to extract names and addresses using spaCy
def extract_names_and_addresses(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    names = []
    addresses = []
    print(">>>>>>>>>>>>>>>")
    print(doc)
    print("extracted_text------------------------------------------------------")
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            names.append(ent.text)
        elif ent.label_ == 'GPE':
            addresses.append(ent.text)
    
    return names, addresses    

if __name__ == '__main__':
    app.run(debug=True)

#if __name__ == "__main__":
    #image_path = "images/DL_fake.jpg"
    #extracted_text = extract_text_from_image(image_path)
    #print("Extracted Text:")
    #print(extracted_text)
    #extracted_text= extract_text_from_pdf("images/Sample_Sales_Order.pdf")
    #print("Extracted Text:")
    #print(extracted_text)
    #name_address= extract_names_and_addresses(extracted_text)
    #print(name_address)
