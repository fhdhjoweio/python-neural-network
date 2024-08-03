import sys
import threading
import numpy as np
import cv2
from PIL import Image
import pytesseract
# from PIL.Image import Image
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QTextCursor
from textblob import TextBlob
import re
import openai
# from textblob import TextBlob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from transformers import pipeline
# from PySide2.QtWidgets import QApplication, QTextEdit
# from PySide2.QtGui import QTextCursor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

openai.api_key = "sk-iAi7372a92266c786b27b151cec19c0f56902308639YPU27"
openai.api_base = "https://api.gptsapi.net/v1"

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class MainApp(QtWidgets.QWidget):
    update_text_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.timer = None
        self.progress_bar = None
        self.text_widget = None
        self.model_combobox = None
        self.upload_button = None
        self.tone_combobox = None
        self.init_ui()
        self.message_queue = []
        self.current_display = ''
        self.char_index = 0

        self.update_text_signal.connect(self.display_text)

    def init_ui(self):
        self.setWindowTitle("Chat Record Extractor")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("QWidget { background-color: #2b2b2b; }")

        layout = QtWidgets.QVBoxLayout()

        # Upload Button
        self.upload_button = QtWidgets.QPushButton(QtGui.QIcon("upload_icon.png"), " Upload Image")
        self.upload_button.setFont(QtGui.QFont("Arial", 12))
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setStyleSheet(
            "QPushButton { background-color: #3c3f41; color: white; border: 2px solid #3c3f41; border-radius: 5px; }"
            "QPushButton:hover { background-color: #4b6eaf; }")
        layout.addWidget(self.upload_button)

        # Tone ComboBox
        self.tone_combobox = QtWidgets.QComboBox()
        self.tone_combobox.addItems(["Automatic", "Positive", "Negative", "Neutral"])
        self.tone_combobox.setStyleSheet(
            "QComboBox { background-color: #3c3f41; color: white; border: 1px solid white; }"
            "QComboBox QAbstractItemView { background-color: #3c3f41; color: white; selection-background-color: #4b6eaf; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox::down-arrow { image: url(down_arrow_icon.png); }"  # 可选，如果你有下拉箭头图标
        )
        layout.addWidget(self.tone_combobox)

        # Model ComboBox
        self.model_combobox = QtWidgets.QComboBox()
        self.model_combobox.addItems(["GPT 3.5", "GPT 2", "Qwen2 (1.5B)", "Qwen2 (7B)", "OPT (125M)", "Phi 3 Mini",])
        self.model_combobox.setStyleSheet(
            "QComboBox { background-color: #3c3f41; color: white; border: 1px solid white; }"
            "QComboBox QAbstractItemView { background-color: #3c3f41; color: white; selection-background-color: #4b6eaf; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox::down-arrow { image: url(down_arrow_icon.png); }"  # 可选，如果你有下拉箭头图标
        )
        layout.addWidget(self.model_combobox)

        # Text Widget
        self.text_widget = QtWidgets.QTextEdit()
        self.text_widget.setReadOnly(True)
        self.text_widget.setFont(QtGui.QFont("Consolas", 10))
        self.text_widget.setStyleSheet(
            "QTextEdit { background-color: white; color: black; border: 1px solid #555; border-radius: 5px; padding: 10px; }")
        layout.addWidget(self.text_widget)

        # Progress Bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { border: 2px solid grey; border-radius: 5px; }"
                                        "QProgressBar::chunk { background-color: #4b6eaf; width: 20px; }")
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.display_next_char)
        self.timer.setInterval(5)

    def upload_image(self):
        print("Attempting to open file dialog...")  # Debug print
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        print("File path:", file_path)  # Debug print
        if file_path:
            self.progress_bar.setVisible(True)  # Indeterminate progress
            threading.Thread(target=self.process_image, args=(file_path,)).start()
        else:
            print("No file selected.")  # Debug print

    def process_image(self, file_path):
        # 处理图片并分析文本
        try:
            chat_texts = extract_and_analyze_text(file_path)
            if self.model_combobox.currentText() == "GPT 3.5":
                response = generate_response_from_right_gpt_openai(self, chat_texts)
            # elif self.model_combobox.currentText() == "GPT 4":
            #     response = generate_response_from_right_gpt_4(self, chat_texts)
            else:
                response = generate_response_from_right_local(self, chat_texts)
            self.progress_bar.setVisible(False)
            self.update_text_signal.emit(response)  # 使用信号发送文本
        except Exception as e:
            print("Failed to process image:", e)  # Stop indeterminate progress

    def display_text(self, text):
        # 主线程中处理文本显示
        self.message_queue.append(text)
        if not self.timer.isActive():  # Start the timer if it's not already running
            self.display_next_message()

    def display_next_message(self):
        # Start displaying next message
        if self.message_queue:
            self.current_display = self.message_queue.pop(0) + ' '  # Add space at end to separate messages
            self.char_index = 0
            self.text_widget.clear()
            self.timer.start()

    def display_next_char(self):
        # Display the next character in the text widget
        if self.char_index < len(self.current_display):
            self.text_widget.insertPlainText(self.current_display[self.char_index])  # Insert next character
            self.char_index += 1
        else:
            self.timer.stop()  # Stop the timer if end of message is reached
            if self.message_queue:  # If there are more messages, start the next one
                self.display_next_message()


def model_define(model_name):
    model_map = {
        "GPT 2": "openai-community/gpt2-large",
        "Qwen2 (1.5B)": "Qwen/Qwen2-1.5B-instruct",
        "Qwen2 (7B)": "Qwen/Qwen2-7B-Instruct",
        "OPT (125M)": "facebook/opt-125m",
        "Phi 3 Mini": "microsoft/Phi-3-mini-4k-instruct"
    }
    return model_map.get(model_name)

# pil_image = None

def extract_and_analyze_text(image_path):
    try:
        pil_image = Image.open(image_path)
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1]  # Convert RGB to BGR

        image = enhance_image(open_cv_image)
        width = image.shape[1]  # Get the width of the image

        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        raw_chat_texts = [
            (clean_text(ocr_data['text'][i]), 'left' if ocr_data['left'][i] < width / 2 else 'right')
            for i in range(len(ocr_data['text']))
                if int(ocr_data['conf'][i]) > 60 and re.search(r'\w', ocr_data['text'][i])
        ]

        last_side = ""
        msg = ""
        chat = []
        for t, s in raw_chat_texts:
            if s == last_side:
                msg += " " + t
            else:
                chat.append((last_side, msg))
                last_side=s
                msg = t
        chat.append((last_side, msg))
        return chat

    except Exception as e:
        print(f"Error occurs: {e}")
        return []


def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh


def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def generate_response_from_right_gpt_openai(self, chat_texts):
    full_text = ' '.join(text for _, text in chat_texts)
    user_tone = self.tone_combobox.currentText()

    if user_tone == "Automatic":
        sentiment = TextBlob(full_text).sentiment.polarity
        tone = 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'
    else:
        tone = user_tone.lower()

    input_text = f"This conversation is {tone}. Your role is to understand what user's statement and generate the reply directly for user. Make sure to limit your answer to 35 tokens. Reply in a way that's more like chatting among friends. "

    prompt = [
            {"role": "system", "content": "You are a helpful assistant. A new conversation is start. "},
            {"role": "user", "content": input_text}
    ]

    for side, msg in chat_texts:
        if side=="left":
            prompt.append({"role": "assistant", "content": msg})
            print(msg + '\n')
        elif side=="right":
            prompt.append({"role": "user", "content": msg})
            print(msg + '\n')

    # prompt.append({"role": "system", "content": input_text})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=50,
        temperature=0.5,
        # top_p=0.3,
        # top_k=10,
        stop=[".", "?", "!", "\n"],
    )
    print(response)
    # print(ensure_complete_sentences(response['choices'][0]['message']['content']))
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']


# def generate_response_from_right_gpt_4(self, chat_texts):
#     full_text = ' '.join(text for _, text in chat_texts)
#     user_tone = self.tone_combobox.currentText()
#
#     if user_tone == "Automatic":
#         sentiment = TextBlob(full_text).sentiment.polarity
#         tone = 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'
#     else:
#         tone = user_tone.lower()
#
#     input_text = f"This conversation is {tone}. Your role is to understand what user's statement and generate the reply directly for user. Make sure to limit your answer to 35 tokens. Reply in a way that's more like chatting among friends. "
#
#     prompt = [
#             {"role": "system", "content": "You are a helpful assistant. A new conversation is start. "},
#             # {"role": "user", "content": input_text}
#     ]
#
#     for side, msg in chat_texts:
#         if side=="left":
#             prompt.append({"role": "assistant", "content": msg})
#         elif side=="right":
#             prompt.append({"role": "user", "content": msg})
#
#     prompt.append({"role": "system", "content": input_text})
#
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=prompt,
#         max_tokens=40,
#         temperature=0.6,
#         top_p=0.3,
#         top_k=10,
#         stop=[".", "?", "!", "\n"],
#     )
#     print(response)
#     # print(ensure_complete_sentences(response['choices'][0]['message']['content']))
#     print(response['choices'][0]['message']['content'])
#     return response['choices'][0]['message']['content']


# def ensure_complete_sentences(text):
#     # 使用正则表达式匹配并分割句子
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     # 移除空句子
#     sentences = [sentence for sentence in sentences if sentence.strip()]
#     # 检查最后一个句子是否以句号、问号或感叹号结束
#     if sentences and not sentences[-1].endswith(('.', '?', '!')):
#         # 如果不是，移除最后一个句子
#         sentences.pop()
#     # 重新组合为一个完整的文本
#     return ' '.join(sentences)


def generate_response_from_right_local(self, chat_texts, model=None):
    if model is None:
        model = model_define(self.model_combobox.currentText())
    full_text = " ".join(text for _, text in chat_texts)

    user_tone = self.tone_combobox.currentText()
    if user_tone == "Automatic":
        sentiment_pipeline = pipeline("sentiment-analysis")
        tone = sentiment_pipeline(full_text)[0]
        if tone["score"] < 0.65:
            tone = "NEUTRAL"
        else:
            tone = tone["label"]
    else:
        tone = user_tone.upper()

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto").to(device)
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model = model.to(device)

    prompt = [
        {"role": "system", "content": f"You are having a conversation in a {tone} tone."},
        {"role": "user", "content": "Generate a response directly"}
    ]

    for side, msg in chat_texts:
        if side=="left":
            prompt.append({"role": "assistant", "content": msg})
        elif side=="right":
            prompt.append({"role": "user", "content": msg})
    input_tokens = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    input_tokens = input_tokens.to(device)

    output_tokens = model.generate(
        input_tokens,
        do_sample=True,
        max_new_tokens=30,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        top_p=0.2,
        top_k=10,
        temperature=0.5,
    )
    output_tokens = [
        output_tokens[len(input_tokens):] for input_tokens, output_tokens in zip(input_tokens, output_tokens)
    ]

    output = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return output

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = MainApp()
    ex.show()
    sys.exit(app.exec_())
