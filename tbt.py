from aiogram.client.default import DefaultBotProperties  # Импорт свойств для настройки бота по умолчанию
from aiogram.enums import ParseMode  # Импорт перечисления режимов парсинга для форматирования сообщений
import logging  # Импорт модуля логирования для отслеживания действий и ошибок в коде
import asyncio  # Импорт asyncio для работы с асинхронными задачами и событиями
from aiogram import Bot, Dispatcher  # Импорт классов бота и диспетчера для работы с Telegram Bot API
from aiogram.types import Message, CallbackQuery, Voice, Document, BotCommand, FSInputFile
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from aiogram.filters import Command  # Импорт фильтра для обработки текстовых команд, таких как /start и /help
from aiogram import F  # Импорт F - для работы с фильтрацией данных сообщений

import pytz
from pytz import tzinfo

from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.retrievers import FAISSRetriever
####   retrieved_docs = retriever.get_relevant_documents(query)
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
import os
import getpass
import re
import requests
import pandas as pd
import csv
import json
import gspread
from langchain.docstore.document import Document
import openai
from openai import OpenAI
import tiktoken
import matplotlib.pyplot as plt
from datetime import datetime


client_oai = OpenAI(api_key="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")
TELEGRAM_TOKEN="7576999190:AAHE1nkJhS1A4kkZqzdph_c5JQihGR54dm0"
LLM_LANGUAGES = {
    "RU": "Русский",
    "EN": "Английский",
    "AR": "Арабский"
}
LLM_LANGUAGE = 'RU'


# Создание экземпляра бота
bot = Bot(token=TELEGRAM_TOKEN)

dp = Dispatcher()

async def setup_main_menu(bot: Bot, commands):
    # Создаем команды для главного меню
    # main_menu_commands = [BotCommand(command=cmd, description=desc) for cmd, desc in commands]
    main_menu_commands = commands


logging.basicConfig(level=logging.INFO)

# Обработчик команды /start
@dp.message(Command('start'))  # Декоратор, который регистрирует хендлер для обработки команды /start
async def cmd_start(message: Message):
    # message - объект класса Message. Содержит всю информацию о сообщении, полученном ботом: текст, отправителя, время отправки и прочее.
    # https://docs.aiogram.dev/en/latest/api/types/message.html

    # Отправляем приветственное сообщение пользователю
    await message.answer("В поле \"Написать сообщение\" наберите вопрос по исламу/Корану и отправим его в ChatGPT...")


# Обработчик команды /help
@dp.message(Command('help'))  # Декоратор, который регистрирует хендлер для обработки команды /help
async def cmd_help(message: Message):
    # Отправляем сообщение с доступными командами и инструкциями для пользователя
    await message.answer("Вот список доступных команд:\n/start - начать работу со мной\n/model - выбор языковой модели (рабоает-GPT-4o-mini)\n/options - выбор языка обработки запросов к LLM (см./about)/help - показать это сообщение помощи\n/about - информация обо мне и настройках\n\nПосле завершения настроек и выбора языка обращения к ChatGPT введите вопрос и нажмите ввод.")


# Обработчик команды /about
@dp.message(Command('about'))  # Декоратор, который регистрирует хендлер для обработки команды /about
async def cmd_about(message: Message):
    # Отправляем информацию о боте, его возможностях и назначении
    await message.answer("Привет! Я бот-нейрособеседник по Корану. Моя миссия - помочь Вам в познании ислама и Корана.\n\nЯ работаю в тестовом режиме: выбор языка предназначен для тестирования модели. На каком бы из 3-х языков вы не задавали вопрос, вопрос к языковой модели будет переведён на выбранный язык. Ответ будет представлен на выбранном языке и переведён на русский\n\nПосле выбора языка обращения к языковой модели введите вопрос и нажмите ввод.")

@dp.startup()
async def set_menu_button(bot: Bot):
    # Определение основных команд для главного меню - кнопка (Menu) слева внизу
    main_menu_commands = [
        BotCommand(command='/start', description='Start'),  # Добавляем команду /start с описанием "Start"
        BotCommand(command='/model', description='Model LLM'),  # Добавляем команду /model с описанием "Model"
        BotCommand(command='/options', description='Options'),  # Добавляем команду /options с описанием "Options"
        BotCommand(command='/help', description='Help Information'),  # Добавляем команду /help с описанием "Help Information"
        BotCommand(command='/about', description='About this bot')]  # Добавляем команду /about с описанием "About this bot"
    await bot.set_my_commands(main_menu_commands)

async def generate_answer(task, query):
    # Определяем системное сообщение, которое задает роль и стиль общения модели
    system_message = {
        "role": "system",
        "content": task
    }

    # Создаем список сообщений, включая системное сообщение и запрос пользователя
    messages = [system_message, {"role": "user", "content": query}]

    # Вызываем API OpenAI для генерации ответа на основе переданных сообщений
    response = client_oai.chat.completions.create(
        model="gpt-4o-mini",  # Указываем модель для генерации ответа
        messages=messages,     # Передаем список сообщений
        temperature=0,      # Устанавливаем параметр температуры для контроля креативности (0 - детерминированный ответ, 1 - более креативный)
        max_tokens=1000       # Указываем максимальное количество токенов в ответе
    )

    # Возвращаем текст ответа от модели
    # return response['choices'][0]['message']['content']
    return response.choices[0].message.content

    # async def generate_answer_own_quran(task, query, dbv):
    #     ans_llm = llm_answer(task, query, dbv)
    #     return ans_llm


# Ловим все текстовые сообщения, кроме "/model" и "/options". Используем MagicFilters - F
@dp.message(F.text & ~(F.text == "/model") & ~(F.text == "/options"))  # Декоратор для регистрации хендлера сообщений, который будет реагировать на текстовые сообщения.
async def handle_text_message(message: Message):  # Определение асинхронной функции-хендлера, принимающей объект сообщения.
    # message - объект класса Message. Содержит всю информацию о сообщении, полученном ботом: текст, отправителя, время отправки и прочее.
    # https://docs.aiogram.dev/en/latest/api/types/message.html

    await message.answer(f"Вы отправили вопрос к LLM: {message.text}\nЯзык обращения к LLM будет: {LLM_LANGUAGES[LLM_LANGUAGE]}")  # Отправка ответа пользователю с текстом его сообщения.

    if (not LLM_LANGUAGE=='RU'):
        if (LLM_LANGUAGE=='EN'):
            task = 'Переведи текст в тройных кавычках на английский язык. Из ответа убери тройные кавычки'
        if (LLM_LANGUAGE=='AR'):
            task = 'Переведи текст в тройных кавычках на арабский язык. Из ответа убери тройные кавычки'

        translate_question = await generate_answer(task, "'''"+message.text+"'''")
        await message.answer(f"Переведённый вопрос: {translate_question}")

    if (LLM_LANGUAGE=='RU'):
        #task = timl_VZ_prmt_0
        #dbv = quran_ru_vektor
        translate_question = message.text
    # else:
    #     if (LLM_LANGUAGE=='EN'):
    #         #task = timl_VZ_prmt_0_en
    #         dbv = quran_en_vektor
    #     elif (LLM_LANGUAGE=='AR'):
    #         task = timl_VZ_prmt_0_ar
    #         dbv = quran_ar_vektor

#    llm_answer = await generate_answer_own_quran(task, translate_question, dbv)

    task = 'Отвечай на только вопрос, который должен быть в контексте ислама и Корана, без интерпретаций и комментариев. Из ответа убери тройные кавычки'
    llm_answer = await generate_answer(task, "'''"+translate_question+"'''")
    await message.answer(f"Полученный ответ: {llm_answer}")

    if (not LLM_LANGUAGE=='RU'):
        task = 'Переведи текст в тройных кавычках на русский язык. Из ответа убери тройные кавычки'
        translate_answer = await generate_answer(task, "'''"+llm_answer+"'''")
        await message.answer(f"Переведённый ответ: {translate_answer}")

    # await message.answer(f"Вы отправили текстовое сообщение: {message.text}")  # Отправка ответа пользователю с текстом его сообщения.
    # await message.answer(f'Ваш Телеграм ID: {message.from_user.id}')  # Отправка сообщения пользователю с его Telegram ID.
    # await message.answer(f'Ваш username: {message.from_user.username}') # Отправка сообщения пользователю с его username в Telegram.


# Нажатие на Reply-кнопку инициирует отправку текста, который был на кнопке,
# в виде обычного текстового сообщения от пользователя боту

# Импортируем класс KeyboardButton с псевдонимом KB для создания кнопок Reply-клавиатуры
from aiogram.types import KeyboardButton as KB


# Функция создания Reply-клавиатуры
async def reply_keyboard():
    # Список списков. Внутренний список - это кнопки в одну строку
    kb = [
           [KB(text="GPT-4o-mini"), KB(text="GPT-4o")],  # Создаем кнопки для первой строки клавиатуры
           [KB(text="Gemini 1.5 Flash"), KB(text="Gemini 1.5 Pro")],  # Создаем кнопки для второй строки
           [KB(text="Llama 3")]  # Создаем кнопку для третьей строки
        ]
    return ReplyKeyboardMarkup(
            keyboard=kb,  # Передаем созданные кнопки в параметр keyboard
            resize_keyboard=True, # Клавиатура будет подстраиваться под размер экрана
            one_time_keyboard=True,  # Клавиатура исчезнет после выбора варианта
            input_field_placeholder="Выберите модель LLM")  # Подсказка для поля ввода


# Функция создания такой же Reply-клавиатуры используя ReplyKeyboardBuilder
async def reply_keyboard_builder():
    # Список названий кнопок
    buttons = ["GPT-4o-mini", "GPT-4o", "Gemini 1.5 Flash", "Gemini 1.5 Pro", "Llama 3"]
    builder = ReplyKeyboardBuilder()  # Создаем экземпляр ReplyKeyboardBuilder для создания клавиатуры
    for element in buttons:
        builder.button(text=element)  # Добавляем кнопки по очереди с текстом из списка
    builder.adjust(2)  # Установка количества кнопок в одной строке
    return builder.as_markup(
                resize_keyboard=True, # Клавиатура будет подстраиваться под размер экрана
                one_time_keyboard=True,  # Клавиатура исчезнет после выбора варианта
                input_field_placeholder="Выберите модель LLM")  # Подсказка для поля ввода


# Обработчик команды /model
@dp.message(Command('model')) # Декоратор, который регистрирует хендлер для обработки команды /model
async def handle_start_command(message: Message):
    # посылаем запрос пользователю и открываем Reply-клавиатуру
    await message.answer("Выберите одну из моделей (доступна только первая):",
                         # reply_markup=await reply_keyboard(),  # Reply-клавиатура, созданная функцией reply_keyboard
                         reply_markup=await reply_keyboard_builder(), # Reply-клавиатура, созданная reply_keyboard_builder
                         )


# Импортируем класс InlineKeyboardButton с псевдонимом IKB для создания кнопок Inline-клавиатуры
from aiogram.types import InlineKeyboardButton as IKB

# Функция создания Inline-клавиатуры
async def inline_keyboard():
    # Список списков. Внутренний список - это кнопки в одну строку
    kb = [
        # Первая строка кнопок
        [IKB(text="Рус.", callback_data='RUS'), # Создаем кнопку с callback_data 'RUS'
         IKB(text="Англ.", callback_data='ENG'), # Создаем кнопку с callback_data 'ENG'
         IKB(text="Араб.", callback_data='ARA')], # Создаем кнопку с callback_data 'ARA'
        # Вторая строка кнопок.   # Создаем кнопку для перехода по URL
        [IKB(text="Перейти на сайт УИИ", url="https://neural-university.ru/")],
        ]
    return InlineKeyboardMarkup(inline_keyboard=kb) # Возвращаем объект Inline-клавиатуру


# Функция создания такой же Inline-клавиатуры используя InlineKeyboardBuilder
async def inline_keyboard_builder():
    builder = InlineKeyboardBuilder() # Создаем экземпляр класса InlineKeyboardBuilder
    builder.button(text="Русский", callback_data='RU')  # Создаем кнопку с callback_data 'RU'
    builder.button(text="Английский", callback_data='EN')  # Создаем кнопку с callback_data 'EN'
    builder.button(text="Арабский", callback_data='AR')  # Создаем кнопку с callback_data 'AR'
    # builder.button(text="Перейти на сайт УИИ", url="https://neural-university.ru/")  # Создаем кнопку для перехода по URL
    builder.adjust(3)  # Устанавливаем количество кнопок в строке (3 кнопки в одной строке)
    return builder.as_markup() # Возвращаем объект Inline-клавиатуру

# Хендлер команды /options для отображения Inline-клавиатуры
@dp.message(Command('options'))  # Декоратор, который регистрирует хендлер для команды /options
async def handle_options_command(message: Message):
    await message.answer("Выберите язык для LLM:", # Отправляем сообщение с клавиатурой
                         reply_markup=await inline_keyboard_builder()) # Получаем созданную Inline-клавиатуру
# Хендлер callback-запросов
@dp.callback_query(F.data)  # Декоратор для дюбых callback запросов
async def handle_callback(callback: CallbackQuery):

    global LLM_LANGUAGE
    LLM_LANGUAGE = callback.data

    if callback.data == 'RU':
        await callback.message.answer("Вы выбрали русский язык обращения к LLM")
    if callback.data == 'EN':
        await callback.message.answer("Вы выбрали английский язык обращения к LLM")
    if callback.data == 'AR':
        await callback.message.answer("Вы выбрали анрабский язык обращения к LLM")
    await callback.message.answer(f"Значение callback.data:  {callback.data}")
    # Убираем Inline-клавиатуру из сообщения
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer()  # Ответ на callback для предотвращения зависания интерфейса

async def main():
    try:
        print("Запуск бота...")
        await dp.start_polling(bot)  # Запускаем процесс polling для получения и обработки обновлений от Telegram
    finally:
        print("Остановка бота...")
        await bot.session.close()  # Закрываем сессию бота для корректного завершения работы и освобождения ресурсов

def run_bot():
  asyncio.run(main())

run_bot()



