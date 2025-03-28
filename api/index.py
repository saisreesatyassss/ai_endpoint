

from flask import Flask, request, jsonify
import re
import io
import json  
from google import genai
import os
from dotenv import load_dotenv  # Import load_dotenv
from flask_cors import CORS  # Import CORS for handling cross-origin requests

# Load environment variables from .env file
load_dotenv()


app = Flask(__name__)

CORS(app)

api_key = os.getenv('GOOGLE_API_KEY')

# Check if the API key exists
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from the environment variables")

client = genai.Client(api_key=api_key)



@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

@app.route('/send_message', methods=['POST'])
def send_message():
    # Get the message from the POST request
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Query the Azure OpenAI GPT model
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Explain how AI works",
        )
          # Print out the response to understand its structure
        print(response)

        # Try to access the correct attribute (check the structure)
        if hasattr(response, 'text'):
            return jsonify({"response": response.text})
        else:
            return jsonify({"error": "Unexpected response format"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


 


@app.route('/full_medical_analysis/<query>', methods=['POST'])
def full_medical_analysis(query):
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Medical history and query are required"}), 400
    
    # Remove '_id' fields recursively
    def remove_ids(d):
        if isinstance(d, dict):
            return {k: remove_ids(v) for k, v in d.items() if k != "_id"}
        elif isinstance(d, list):
            return [remove_ids(i) for i in d]
        return d
    
    cleaned_data = remove_ids(data)
    medical_history = cleaned_data
    
    # Format the input for AI processing
    prompt = f"""
    You are an advanced medical AI chatbot. Analyze the provided medical history and respond accurately.
    
    Patient Medical History:
    {medical_history}
    
    User Query:
    {query}
    
    Provide a medically accurate and insightful response based on the given history.
    Return your response in JSON format with the following structure:
    {{
        "summary": "Brief 1-2 sentence summary",
        "details": [
            {{
                "category": "category name (e.g. Cardiovascular, Allergies, Medications)",
                "findings": ["finding 1", "finding 2", "..."]
            }}
        ],
        "recommendations": ["recommendation 1", "recommendation 2", "..."]
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        if hasattr(response, 'text'):
            # Process the response - attempt to parse it as JSON
            try:
                # First, try direct JSON parsing
                import json
                response_text = response.text.strip()
                structured_data = json.loads(response_text)
                return jsonify(structured_data)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the text
                # This handles cases where the model might add extra text
                import re
                json_match = re.search(r'({.*})', response.text.replace('\n', ' '), re.DOTALL)
                if json_match:
                    try:
                        structured_data = json.loads(json_match.group(1))
                        return jsonify(structured_data)
                    except:
                        pass
                
                # Parse Markdown list items and create a simple structure
                lines = response.text.strip().split('\n')
                
                # Simple structure detection
                summary = lines[0] if lines else ""
                details = []
                
                current_category = "General"
                current_findings = []
                
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if this is a category header
                    if line.startswith('**') and line.endswith(':**'):
                        # Save previous category if it has findings
                        if current_findings:
                            details.append({
                                "category": current_category,
                                "findings": current_findings
                            })
                            
                        # Start new category
                        current_category = line.strip('*: ')
                        current_findings = []
                    # Check if this is a list item
                    elif line.startswith('*') or line.startswith('-'):
                        item = line[1:].strip()
                        current_findings.append(item)
                
                # Add the last category
                if current_findings:
                    details.append({
                        "category": current_category,
                        "findings": current_findings
                    })
                
                # Create structured response
                structured_response = {
                    "summary": summary,
                    "details": details,
                    "recommendations": []  # Add logic to extract recommendations if available
                }
                
                return jsonify(structured_response)
        else:
            return jsonify({"error": "Unexpected response format"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
     
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
