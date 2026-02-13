import os
import json
from flask import Flask, request, jsonify
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)

# ---------------------------------------------------
# Configure Gemini
# ---------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ---------------------------------------------------
# STEP 1: Read HTML template ONLY ONCE at startup
# ---------------------------------------------------
with open("hrms_template.html", "r", encoding="utf-8") as f:
    HRMS_TEMPLATE = f.read()

print("✅ HTML template loaded once at startup")

# ---------------------------------------------------
# API ENDPOINT
# ---------------------------------------------------
@app.route('/extract', methods=['POST'])
def extract_fields():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Image file required"}), 400

        image_file = request.files['image']

        if image_file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        img = Image.open(image_file.stream)

        # ---------------------------------------------------
        # STEP 2: Extract text from image (OCR)
        # ---------------------------------------------------
        extraction_prompt = "Extract ALL visible text from this image. Return only plain text."

        extraction_response = model.generate_content([extraction_prompt, img])
        extracted_text = extraction_response.text

        # ---------------------------------------------------
        # STEP 3: Convert OCR text into structured JSON
        # HTML is used ONLY as layout context
        # ---------------------------------------------------
        json_prompt = f"""
You are given:

1) HTML template of an HRMS page (context only):
--------------------------
{HRMS_TEMPLATE[:3000]}
--------------------------

IMPORTANT:
The HTML is ONLY for understanding page structure.
DO NOT extract any values from it.

2) OCR text extracted from screenshot:
--------------------------
{extracted_text}
--------------------------

TASK:
From the OCR text, identify meaningful key-value pairs.

Rules:
- Extract only real HR data fields
- Ignore menus, buttons, sidebar labels
- Detect patterns like "Label : Value"
- Return ONLY valid JSON
- No explanation
"""

        json_response = model.generate_content(json_prompt)

        output = json_response.text.strip()

        # Remove markdown formatting if present
        output = output.replace("```json", "").replace("```", "").strip()

        # Convert to real JSON object safely
        try:
            parsed_json = json.loads(output)
        except:
            parsed_json = {"raw_output": output}

        return jsonify(parsed_json)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
