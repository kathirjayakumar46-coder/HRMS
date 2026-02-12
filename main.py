import os
from flask import Flask, request, Response
import google.generativeai as genai
from PIL import Image
import io

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")    #loads from .env

genai.configure(api_key=GEMINI_API_KEY)    #configure api_key

model = genai.GenerativeModel("gemini-2.0-flash")   #name of the model


@app.route('/ask', methods=['POST'])       #flask_app post method
def ask_question():
    try:
        question = request.form.get('question')
        
        if not question:                       #question is mandatory if not raise error
            return Response("Error: Question is required", status=400, mimetype='text/plain')
        
        if 'image' not in request.files:        #image is mandatory if not raise error
            return Response("Error: Image file is required", status=400, mimetype='text/plain')
        
        image_file = request.files['image']      #request to open the image
        
        if image_file.filename == '':             #if there is no text in image raise error
            return Response("Error: No image selected", status=400, mimetype='text/plain')
        
        img = Image.open(image_file.stream)       # stream is used for text animation
        
        extraction_prompt = "Extract ALL visible text from this image. Return only the text content, no additional commentary."
        
        extraction_response = model.generate_content([extraction_prompt, img])
        
        extracted_text = extraction_response.text
        
        answer_prompt = f"""
Based on the following text extracted from the image:

{extracted_text}

Answer this question: {question}

If the answer is not present in the text, respond exactly:
Not found in the image.
"""
        
        def generate():
            response_stream = model.generate_content(answer_prompt, stream=True)
            for chunk in response_stream:
                text = getattr(chunk, "text", "")
                for char in text:
                    yield char
        
        return Response(generate(), mimetype='text/plain')
        
    except Exception as e:
        return Response(f"Error: {str(e)}", status=500, mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)