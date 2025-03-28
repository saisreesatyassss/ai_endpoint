from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS
from google import genai
import replicate

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv('GOOGLE_API_KEY')
replicate_api_token = os.getenv('REPLICATE_API_TOKEN')

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from the environment variables")

if not replicate_api_token:
    raise ValueError("REPLICATE_API_TOKEN is missing from the environment variables")

client = genai.Client(api_key=api_key)

@app.route('/')
def home():
    return 'API is running!'

@app.route('/google', methods=['POST'])
def google_generate():
    data = request.json
    user_input = data.get('text', '')

    if not user_input:
        return jsonify({"error": "Text input is required"}), 400

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_input,
        )
        
        if hasattr(response, 'text'):
            return jsonify({"response": response.text})
        else:
            return jsonify({"error": "Unexpected response format"}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/snowflake', methods=['POST'])
def snowflake_generate():
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        # Initialize Replicate client
        replicate_client = replicate.Client(api_token=replicate_api_token)
        
        # Define model and parameters
        output = ""
        for event in replicate_client.stream(
            "snowflake/snowflake-arctic-instruct",  # Replace with actual model ID if needed
            {"prompt": prompt, "temperature": 0.75, "max_new_tokens": 500}
        ):
            if isinstance(event, str):  # Ensure data is correctly extracted
                output += event
        
        return jsonify({"response": output})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)