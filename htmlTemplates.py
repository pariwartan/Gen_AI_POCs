css = '''
<style>
div[class*="stTextInput"] label p{
  font-size: 25px;
  font-weight: bold;
  color: white;
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: black
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.mywisely.com/wp-content/uploads/2022/08/logo-wisely-color.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.mywisely.com/wp-content/uploads/2022/07/female_holding_phone_waves_background.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

page_bg_img = '''
<style>
.stApp {
background-image: url("https://www.mywisely.com/wp-content/uploads/2022/07/wisely_floatingcards.png" );
background-size: cover;
background-color: #6420ed;
}
</style>
'''

