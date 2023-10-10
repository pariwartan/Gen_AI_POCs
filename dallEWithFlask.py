import openai
from flask import Flask, render_template, request, redirect, url_for
from constants import openai_key
import requests
from PIL import Image
import base64
import io

# Replace 'YOUR_API_KEY' with your OpenAI API key
openai.api_key = openai_key

app = Flask(__name__)
image_map={}
#images = []
image_uri=[]
def generate_images(prompt):
    # Request images from DALL·E using the user's input prompt
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256",
    )
    images = response["data"]
    image_uri.clear()
    i = 0
    for image in images:
        image_map[i] = image["url"]
        image_uri.append(image["url"])
        i = i+1
    return image_uri

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        prompt = request.form['prompt']
        image_uri=[]
        images = generate_images(prompt)       
        return render_template('index.html', show_hidden=False, images=images)
    else:
        return render_template('index.html')

@app.route('/select_image/<int:image_id>')
def select_image(image_id):
    # You can use the 'image_id' parameter to perform actions based on the selected image
    print(f"Selected image ID: {image_id}")
    my_res = requests.get(image_map[image_id])

    img1 = Image.open(io.BytesIO(my_res.content))
    img2 = Image.open("images/wisely_card.jpg")
    background = img1.convert("RGBA").resize((450,280))
    overlay = img2.convert("RGBA").resize((450,280))
    
    new_img = Image.blend(background, overlay, 0.45)
    new_img = new_img.convert("RGB")
    data = io.BytesIO()
    new_img.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template('index.html',  images=image_uri, show_hidden=True, user_image=encoded_img_data.decode('utf-8'))
    #return f"Selected image ID: {image_id}"

if __name__ == '__main__':
    app.run(debug=True)
