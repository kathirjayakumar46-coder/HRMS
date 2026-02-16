import io
import os
import json
from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

from utils import clean_html, chunk_text
from vector_store import VectorStore

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY","devkey")

MAX_FILE_SIZE = 5 * 1024 * 1024

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


# -------------------------------------------------
# GLOBAL SESSION STORE
# -------------------------------------------------
VECTOR_STORE = None
CURRENT_TEXT = None


# =================================================
# HELPER → EXTRACT VALUE USING GEMINI
# =================================================
def extract_value(context, query):

    prompt = f"""
Extract the exact value for "{query}" from text below.

Return JSON only:
{{
"field":"{query}",
"value":""
}}

Text:
{context}
"""

    response = gemini_model.generate_content(prompt)

    clean = response.text.replace("```json","").replace("```","").strip()

    try:
        return json.loads(clean)
    except:
        return {"field": query, "value": ""}



# =================================================
# 1️⃣ UPLOAD HTML → BUILD INDEX
# =================================================
@app.route("/upload-html", methods=["POST"])
def upload_html():

    global VECTOR_STORE, CURRENT_TEXT

    if "file" not in request.files:
        return jsonify({"error":"HTML file required"}),400

    file = request.files["file"]
    data = file.read()

    if len(data) > MAX_FILE_SIZE:
        return jsonify({"error":"File too large"}),400

    try:
        html_string = data.decode("utf-8")
    except:
        return jsonify({"error":"Invalid encoding"}),400

    text = clean_html(html_string)

    if not text:
        return jsonify({"error":"No readable text found"}),400

    chunks = chunk_text(text)

    store = VectorStore()
    store.create_index(chunks)

    VECTOR_STORE = store
    CURRENT_TEXT = text

    return jsonify({
        "status":"Document indexed",
        "chunks":len(chunks)
    })


# =================================================
# 2️⃣ ASK QUESTION → SEARCH EXISTING INDEX
# =================================================
@app.route("/ask", methods=["POST"])
def ask():

    global VECTOR_STORE

    if VECTOR_STORE is None:
        return jsonify({"error":"Upload document first"}),400

    if "query" not in request.form:
        return jsonify({"error":"Query required"}),400

    query = request.form["query"].strip().lower()

    results = VECTOR_STORE.search(query, top_k=3)

    if not results:
        return jsonify({"error":"No relevant text found"}),404

    context = "\n".join(results)

    answer = extract_value(context, query)

    return jsonify(answer)



# =================================================
# OPTIONAL IMAGE UPLOAD ROUTE
# =================================================
@app.route("/upload-image", methods=["POST"])
def upload_image():

    global VECTOR_STORE

    if "file" not in request.files:
        return jsonify({"error":"Image required"}),400

    file = request.files["file"]
    data = file.read()

    try:
        image = Image.open(io.BytesIO(data))
    except:
        return jsonify({"error":"Invalid image"}),400

    response = gemini_model.generate_content([
        "Extract all visible text from this image",
        image
    ])

    text = response.text.strip()

    if not text:
        return jsonify({"error":"No text detected"}),400

    chunks = chunk_text(text)

    store = VectorStore()
    store.create_index(chunks)

    VECTOR_STORE = store

    return jsonify({
        "status":"Image text indexed",
        "chunks":len(chunks)
    })


# =================================================
if __name__ == "__main__":
    app.run(debug=True)
