import logging
import os
import textwrap
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
import telegram
from deep_translator import GoogleTranslator


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


text_generator = pipeline("text-generation", model="gpt2", max_length=100)  # Установка max_length


def start(update, context):
    update.message.reply_text('Привет! Отправь мне изображение, и я добавлю к нему текст.')


def image_handler(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    file_path = os.path.join(os.getcwd(), 'user_photo.jpg')
    photo_file.download(file_path)


    img = Image.open(file_path)


    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), img.size], fill="black")


    inputs = processor(images=img, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)


    prompt = f"{caption}"
    funny_text = text_generator(prompt)[0]['generated_text']  # Убраны параметры max_length и num_return_sequences
    ans_rus = GoogleTranslator(source='en', target='ru').translate(funny_text)
    funny_text = ans_rus

    wrapped_text = textwrap.fill(funny_text, width=30)


    font_size = 40
    font_path = "arial.ttf"  # путь к файлу шрифта
    font = ImageFont.truetype(font_path, font_size)

    y_offset = 50
    for line in wrapped_text.split('\n'):
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((img.width - text_width) / 2, y_offset)

        draw.text(position, line, (255, 255, 255), font=font)
        y_offset += text_height + 10


    meme_photo_path = os.path.join(os.getcwd(), 'meme_photo.jpg')
    img.save(meme_photo_path)


    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(meme_photo_path, 'rb'))


    os.remove(file_path)
    os.remove(meme_photo_path)


def error(update, context):
    logger.warning(f'Update "{update}" caused error "{context.error}"')

def main():

    TOKEN = '7109208236:AAEUuq8sLHbgC2CVT5xP1defwGrI6FdMTnk'
    bot = telegram.Bot(token=TOKEN)
    updater = Updater(bot=bot, use_context=True)  # Удалите max_connections
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.photo, image_handler))

    dispatcher.add_error_handler(error)

    # Запускаем бота
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
