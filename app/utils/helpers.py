import requests
import os

# Read prompt file
def get_prompt(prompt_filename: str) -> str | None:
    try:
        with open(f"prompts/{prompt_filename}", "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None

# Function to call Gemini API with input data and prompt
def get_step_by_step_solution(input_data: str, prompt_filename: str, is_text: bool = False) -> str:
    prompt = get_prompt(prompt_filename)
    
    if not prompt:
        return "Error: Could not read the prompt file."

    headers = {
        "Authorization": f"Bearer {os.getenv('GEMINI_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
    }

    if is_text:
        data["text"] = input_data
    else:
        data["file_path"] = input_data  # Ensure backend knows how to handle this

    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            return response.json().get("solution", "No solution found.")
        else:
            print(f"Gemini API Error (Status {response.status_code}): {response.text}")
            return f"Error processing request: {response.text}"

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return "Error processing the request, please check the connection and API configuration."
