from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import fitz
import google.generativeai as genai
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from datetime import datetime
import base64
from dotenv import load_dotenv
import io
from typing import Optional
import random

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Doubt Solver API")

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable not set")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize MongoDB connection with Motor (async driver)
client = AsyncIOMotorClient(MONGODB_URI)
db = client["doubt_solver"]
# Use Motor's async GridFS implementation
fs_bucket = AsyncIOMotorGridFSBucket(db)

# Create collections
uploads_collection = db.uploads
solutions_collection = db.solutions
conversation_collection = db.conversations

# Pydantic models
class TextRequest(BaseModel):
    text: str

class SolutionResponse(BaseModel):
    solution: str
    
@app.on_event("startup")
async def startup_db_client():
    try:
        await client.admin.command('ping')
        print("Connected to MongoDB!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        
@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    print("MongoDB connection closed")

# List of follow-up questions based on subject areas
follow_up_questions = {
    "math": [
        "Would you like to see a different approach to solve this problem?",
        "Do you understand how we applied the integration formula here?",
        "Should I explain any specific step in more detail?",
        "Would you like to try a similar problem to practice this concept?",
        "Is there a specific part of the solution that's confusing you?"
    ],
    "physics": [
        "Do you understand how we applied conservation of energy in this problem?",
        "Would you like me to explain the free-body diagram in more detail?",
        "Should we go through another example to reinforce this concept?",
        "Is there a specific formula or principle you'd like me to explain further?",
        "Would you like to see a real-world application of this concept?"
    ],
    "chemistry": [
        "Do you understand how we balanced this chemical equation?",
        "Would you like me to explain the mechanism of this reaction in more detail?",
        "Should I clarify anything about the molecular structure?",
        "Would you like to see how this concept applies in laboratory settings?",
        "Is there anything specific about the periodic trends that's unclear?"
    ],
    "general": [
        "Do you have any other questions about this topic?",
        "Would you like me to explain anything else?",
        "Is there a specific part of the explanation that wasn't clear?",
        "Would you like to explore this concept further?",
        "Should I give you some practice problems to test your understanding?"
    ]
}

# Helper function to generate solution using Gemini API
async def generate_solution(prompt, file_content=None, file_type=None, subject_hint="general"):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare content parts based on what's available
        content_parts = [prompt]
        
        # Add file content if available
        if file_content and file_type:
            if file_type.startswith('image/'):
                # For images, we need to handle them differently
                content_parts.append({
                    "mime_type": file_type,
                    "data": base64.b64encode(file_content).decode('utf-8')
                })
            else:
                # For text documents, just add the content
                if isinstance(file_content, bytes):
                    try:
                        text_content = file_content.decode('utf-8')
                        content_parts.append(f"Document content: {text_content}")
                    except UnicodeDecodeError:
                        content_parts.append("Unable to decode document content")
                else:
                    content_parts.append(f"Document content: {file_content}")
        
        # Add instructions to include follow-up question at the end
        if "math" in prompt.lower() or "calculate" in prompt.lower() or "equation" in prompt.lower() or "integral" in prompt.lower():
            subject = "math"
        elif "physics" in prompt.lower() or "force" in prompt.lower() or "motion" in prompt.lower() or "energy" in prompt.lower():
            subject = "physics"
        elif "chemistry" in prompt.lower() or "reaction" in prompt.lower() or "molecule" in prompt.lower() or "compound" in prompt.lower():
            subject = "chemistry"
        else:
            subject = "general"
            
        # Select a random follow-up question from the appropriate category
        follow_up = random.choice(follow_up_questions[subject])
        
        # Add instruction to include follow-up question
        enhanced_prompt = f"{prompt}\n\nAfter providing the solution, end with a natural follow-up question like: '{follow_up}'"
        content_parts[0] = enhanced_prompt
        
        response = model.generate_content(
            content_parts,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 2048,
            }
        )
        
        # Store the solution in MongoDB
        solution_doc = {
            "prompt": prompt,
            "solution": response.text,
            "timestamp": datetime.now(),
            "subject": subject
        }
        await solutions_collection.insert_one(solution_doc)
        
        return response.text
    except Exception as e:
        print(f"Error generating solution: {e}")
        return f"Sorry, I couldn't generate a solution. Error: {str(e)}"

@app.get("/")
async def root():
    return {"message": "Welcome to the Doubt Solver API"}

@app.post("/solve-text")
async def solve_text(request: TextRequest):
    try:
        prompt = f"Explain the solution in a clear, step-by-step manner. Start by identifying what is given and what needs to be found. Then outline the method or concept used to solve it. Solve each step logically, using correct academic notation and terminology (e.g., x², ∫, Δt, moles, sin(θ), etc.), and avoid unnecessary special characters or HTML tags. Keep the explanation structured, not too long, not too short, and conclude with the final answer in a complete sentence. {request.text}"
        solution = await generate_solution(prompt)
        
        return {"solution": solution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read file content
        content = await file.read()
        
        # Store file in MongoDB GridFS using async GridFSBucket
        file_id = await fs_bucket.upload_from_stream(
            file.filename,
            io.BytesIO(content),
            metadata={"content_type": file.content_type}
        )
        
        # Store file metadata
        file_metadata = {
            "grid_fs_id": str(file_id),
            "filename": file.filename,
            "content_type": file.content_type,
            "timestamp": datetime.now()
        }
        
        await uploads_collection.insert_one(file_metadata)
        
        extracted_text = ""
        if file.content_type == "application/pdf":
            extracted_text = await extract_text_from_pdf(content)
        elif file.content_type.startswith("text/"):
            extracted_text = content.decode("utf-8")
        else:
            extracted_text = "Unsupported file type for text extraction."

        # Generate solution based on file content
        prompt = f"Please provide a step-by-step solution for this document:\n{extracted_text}"
        solution = await generate_solution(prompt, content, file.content_type)
        
        return {"solution": solution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text += page.get_text()
                print(f"Page {page_num} extracted text: {text[:200]}")  # Debugging: check extracted text from each page
    except Exception as e:
        print(f"PDF extraction error: {e}")
        text = "Failed to extract text from the PDF."
    return text
       
@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Check if the file is actually an image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image content
        image_content = await image.read()
        
        # Store image in MongoDB GridFS using async GridFSBucket
        file_id = await fs_bucket.upload_from_stream(
            image.filename,
            io.BytesIO(image_content),
            metadata={"content_type": image.content_type}
        )
        
        # Store image metadata
        image_metadata = {
            "grid_fs_id": str(file_id),
            "filename": image.filename,
            "content_type": image.content_type,
            "timestamp": datetime.now()
        }
        
        await uploads_collection.insert_one(image_metadata)
        
        # Generate solution based on image
        prompt = "Please analyze the provided image and Explain the solution in a clear, step-by-step manner. Start by identifying what is given and what needs to be found. Then outline the method or concept used to solve it. Solve each step logically, using correct academic notation and terminology (e.g., x², ∫, Δt, moles, sin(θ), etc.), and avoid unnecessary special characters or HTML tags. Keep the explanation structured, not too long, not too short, and conclude with the final answer in a complete sentence."
        solution = await generate_solution(prompt, image_content, image.content_type, "math")
        return {"solution": solution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send-voice")
async def send_voice(voice: UploadFile = File(...)):
    try:
        # Read voice content
        voice_content = await voice.read()
        
        # Store voice in MongoDB GridFS using async GridFSBucket
        file_id = await fs_bucket.upload_from_stream(
            voice.filename,
            io.BytesIO(voice_content),
            metadata={"content_type": voice.content_type}
        )
        
        # Store voice metadata
        voice_metadata = {
            "grid_fs_id": str(file_id),
            "filename": voice.filename,
            "content_type": voice.content_type,
            "timestamp": datetime.now()
        }
        
        await uploads_collection.insert_one(voice_metadata)
        
        # For voice input, placeholder response
        # In a production environment, you would integrate with a speech-to-text service
        solution = "I've received your voice input. Currently, the system is using a placeholder response. In a production environment, we would use speech-to-text conversion to process your voice input. Explain the solution in a clear, step-by-step manner. Start by identifying what is given and what needs to be found. Then outline the method or concept used to solve it. Solve each step logically, using correct academic notation and terminology (e.g., x², ∫, Δt, moles, sin(θ), etc.), and avoid unnecessary special characters or HTML tags. Keep the explanation structured, not too long, not too short, and conclude with the final answer in a complete sentence.\n\nDo you have any specific math or science problems you'd like me to solve for you?"
        return {"solution": solution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Improved solve-image-text endpoint to better handle image+text combinations
@app.post("/solve-image-text")
async def solve_image_text(
    image: UploadFile = File(...),
    query: str = Form(default="")  # Changed to default="" to make it optional
):
    try:
        # Debug output
        print(f"Received image: {image.filename}, content type: {image.content_type}")
        print(f"Received query: {query}")
        
        # Check file is an image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image content
        image_content = await image.read()
        
        # Save to MongoDB
        file_id = await fs_bucket.upload_from_stream(
            image.filename,
            io.BytesIO(image_content),
            metadata={"content_type": image.content_type}
        )

        await uploads_collection.insert_one({
            "grid_fs_id": str(file_id),
            "filename": image.filename,
            "content_type": image.content_type,
            "query": query,
            "timestamp": datetime.now()
        })

        # Improved prompt handling for different scenarios
        base_prompt = "Please analyze the provided image and explain the solution in a clear, step-by-step manner. Start by identifying what is given and what needs to be found. Then outline the method or concept used to solve it. Solve each step logically, using correct academic notation and terminology (e.g., x², ∫, Δt, moles, sin(θ), etc.), and avoid unnecessary special characters or HTML tags. Keep the explanation structured, not too long, not too short, and conclude with the final answer in a complete sentence."
        
        # If query exists, append it to the prompt
        if query.strip():
            prompt = f"{base_prompt} The user has asked: {query}"
        else:
            # This ensures image-only submissions work properly
            prompt = f"{base_prompt} Solve the problem shown in this image completely."

        # Determine subject from image/query context for better follow-up questions
        subject_hint = "math"  # Default for mathematical images
        if "chemistry" in query.lower() or "molecule" in query.lower() or "reaction" in query.lower():
            subject_hint = "chemistry"
        elif "physics" in query.lower() or "force" in query.lower() or "motion" in query.lower():
            subject_hint = "physics"

        # Generate answer from Gemini with image and text
        solution = await generate_solution(prompt, image_content, image.content_type, subject_hint)

        return {"solution": solution}
    except Exception as e:
        print(f"Error in solve-image-text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# This can be used for testing the API without a file upload
@app.post("/test-query")
async def test_query(query: str = Form(...)):
    return {"received_query": query}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)