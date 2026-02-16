import re
from bs4 import BeautifulSoup


def clean_html(html):

    soup = BeautifulSoup(html,"html.parser")

    for tag in soup(["script","style","meta","link"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+"," ",text).strip()

    return text


def chunk_text(text,size=500,overlap=50):

    if not text:
        return []

    chunks=[]
    start=0

    while start < len(text):
        end=start+size
        chunks.append(text[start:end])
        start += size-overlap

    return chunks
