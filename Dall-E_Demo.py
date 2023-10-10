import openai
from flask import Flask, render_template, request,session, redirect, url_for
from constants import openai_key
import requests
from PIL import Image, ImageDraw, ImageFont
import base64
import io
import os
from rembg import remove


# Replace 'YOUR_API_KEY' with your OpenAI API key
openai.api_key = openai_key

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.secret_key = "super secret key"
image_map={}
#images = []
image_uri=[]
def generate_images(prompt):
    # Request images from DALL·E using the user's input prompt

    response = openai.Image.create(
        prompt=prompt,
        n=5,
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
def start():
    if request.method == "POST":
        return render_template('index_new.html')
    else:   
        return render_template('start.html')
    
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        return render_template('index_new.html')
    else:   
        return render_template('start.html')    
    
@app.route('/success', methods=['GET', 'POST'])
def success():
        print("================================================")
        if request.method == "POST":
            return render_template('success.html')
        else:
            return render_template('success.html')              

@app.route('/submit', methods=['GET', 'POST'])
def enrollment():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       first_name = request.form.get("firstname")
       # getting input with name = lname in HTML form
       last_name = request.form.get("lastname")
       cardName = first_name +" " + last_name
       session['cardName']=cardName
    return render_template("enrollment.html", cardName=cardName)

    
@app.route('/gen_image', methods=['GET', 'POST'])
def gen_image():
    if request.method == 'POST':
        prompt = request.form['prompt']
        image_uri=[]
        images = generate_images(prompt)      
        return render_template('index_new.html', show_hidden=False, images=images)
    else:
        return render_template('index_new.html')

@app.route('/gen_bg', methods=['GET', 'POST'])
def gen_bg():
    if request.method == 'POST':
        prompt = request.form['prompt']

        if 'file' not in request.files:
            return redirect(request.url)
    
        file = request.files['file']


        if file.filename == '':
            return redirect(request.url)
    
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        file.save(filename)

        input = Image.open(filename)
        output = remove(input)
        output.save(filename, format='png')
        
        session['personalImage']=filename

        image_uri=[]
        images = generate_images(prompt)      
        return render_template('index_final.html', images=images , show_hidden=False)
    else:
        return render_template('index_final.html')         

@app.route('/select_image/<int:image_id>')
def select_image(image_id):
    # You can use the 'image_id' parameter to perform actions based on the selected image
    print(f"Selected image ID: {image_id}")
    my_res = requests.get(image_map[image_id])

    img1 = Image.open(io.BytesIO(my_res.content))

    img2 = Image.open("images/wisely_card_new.jpg")
    # Create a drawing object
    draw = ImageDraw.Draw(img2)

    # Define the text to add
    text = session['cardName']

    # Specify the font and size
    font = ImageFont.truetype('arial.ttf', size=36)

    # Specify the position where you want to add the text (x, y)
    position = (50, 230)

    # Specify the text color
    text_color = (255, 255, 255)  # RGB color code for white

    # Add the text to the image
    draw.text(position, text, fill=text_color, font=font)

    background = img1.convert("RGBA").resize((450,280))
    overlay = img2.convert("RGBA").resize((450,280))
    
    new_img = Image.blend(background, overlay, 0.45)
    new_img = new_img.convert("RGB")
    data = io.BytesIO()
    new_img.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template('index_new.html',  images=image_uri, show_hidden=True, user_image=encoded_img_data.decode('utf-8'))
    #return f"Selected image ID: {image_id}"

@app.route('/personal_image/<int:image_id>')
def personal_image(image_id):

    # You can use the 'image_id' parameter to perform actions based on the selected image
    print(f"Selected image ID: {image_id}")
    my_res = requests.get(image_map[image_id])

    image1 = Image.open(io.BytesIO(my_res.content))
    image1 = image1.convert("RGBA").resize((450,280))

    filename=session['personalImage']

    personalImage = Image.open(filename)
    personalImage = personalImage.convert("RGBA").resize((450,280))

    image1.paste(personalImage,(0,0),personalImage)
    image1.save("static/images/temp.png")

    img1 = Image.open("static/images/temp.png")

    img2 = Image.open("images/wisely_card_new.jpg")
    # Create a drawing object
    draw = ImageDraw.Draw(img2)

    # Define the text to add
    text = session['cardName']

    # Specify the font and size
    font = ImageFont.truetype('arial.ttf', size=36)

    # Specify the position where you want to add the text (x, y)
    position = (50, 230)

    # Specify the text color
    text_color = (255, 255, 255)  # RGB color code for white

    # Add the text to the image
    draw.text(position, text, fill=text_color, font=font)

    background = img1.convert("RGBA").resize((450,280))
    overlay = img2.convert("RGBA").resize((450,280))
    
    new_img = Image.blend(background, overlay, 0.45)
    new_img = new_img.convert("RGB")
    data = io.BytesIO()
    new_img.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template('index_final.html', images=image_uri , show_hidden=True, user_image=encoded_img_data.decode('utf-8'))
    #return f"Selected image ID: {image_id}"


@app.route('/imageUpload', methods=['GET', 'POST'])
def imageUpload():
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
