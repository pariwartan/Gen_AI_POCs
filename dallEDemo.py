import streamlit as st
from st_click_detector import click_detector
from PIL import Image
import requests
import os
import openai
import requests
import json
from constants import openai_key
#from st_clickable_images import clickable_images


openai.api_key=openai_key

input_text = st.text_input("select customized image for your card:")

image_map = {}
image_content=""
image_uri=[]
if input_text:
   PROMPT =  input_text
   response1 = '{"created": 1668073562,"data": [{"url": "https://images.unsplash.com/photo-1518727818782-ed5341dbd476?w=700"},{"url": "https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=200"},{"url": "https://images.unsplash.com/photo-1582550945154-66ea8fff25e1?w=700"}]}'
   response = openai.Image.create(
        prompt=PROMPT,
        n=1,
        size="256x256",
        )
   images1 =json.loads(response1)["data"]
   #print(images)
   images = response["data"]
   print("Dalle images =======================================")
   print(images)
   i = 0
   for image in images:
    image_map["image"+str(i)] = image["url"]
    image_content +="<a href='#' name='image1' id='"+ "image"+str(i)+"' ><img width='20%' src='"+image["url"]+"'></a>"
    image_uri.append(image["url"])
    i = i+1
   for image in images1:
    image_map["image"+str(i)] = image["url"]
    image_content +="<a href='#' name='image1' id='"+ "image"+str(i)+"' ><img width='20%' src='"+image["url"]+"'></a>"
    i = i+1


content = """
    {{IMAGES}}
    """.replace("{{IMAGES}}", image_content)

"""clicked = clickable_images(
    image_uri,
    titles=[f"Image #{str(i)}" for i in range(5)],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
)"""


#print(content)
my_url = "https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=200"

#print(click_detector(content))
clicked = click_detector(content)

clicked = str(clicked)
if clicked !="-1":
    #clicked = str(clicked)
    st.text("This image got clicked: "+clicked)
    #img1 = Image.open("images/dhruva.jpg")
    # Download the image using requests
    my_res = requests.get(image_map[clicked])

    # Open the downloaded image in PIL
    #my_img = Image.open(BytesIO(my_res.content))
    img1 = Image.open(BytesIO(my_res.content))
    img2 = Image.open("images/wisely_card.jpg")
    background = img1.convert("RGBA").resize((500,300))
    overlay = img2.convert("RGBA").resize((500,300))
    
    new_img = Image.blend(background, overlay, 0.25)
    st.image(new_img)

#st.markdown(f"**{clicked} clicked**" if clicked != "" else "**No click**")