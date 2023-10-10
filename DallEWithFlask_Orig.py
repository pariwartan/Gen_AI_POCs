from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template
import base64
import io


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Open an image file
    image = Image.open("images/wisely_card_new.jpg")
    image2 = Image.open("images/cat_image.jpg")

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the text to add
    text = "Hello, Pillow!"

    # Specify the font and size
    font = ImageFont.truetype('arial.ttf', size=36)

    # Specify the position where you want to add the text (x, y)
    position = (50, 230)

    # Specify the text color
    text_color = (255, 255, 255)  # RGB color code for white

    # Add the text to the image
    draw.text(position, text, fill=text_color, font=font)

    background = image.convert("RGBA").resize((450,280))
    overlay = image2.convert("RGBA").resize((450,280))

    new_img = Image.blend(background, overlay, 0.45)
    new_img = new_img.convert("RGB")
    data = io.BytesIO()
    new_img.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())


    return render_template('final.html', user_image=encoded_img_data.decode('utf-8'))


if __name__ == '__main__':
    app.run(debug=True)

