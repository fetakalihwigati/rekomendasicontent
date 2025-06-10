
import re

def clean_caption(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www.\\S+", "", text)
    text = re.sub(r"[^\\w\\s]", "", text)
    text = re.sub(r"\\n", " ", text)
    return text
