import streamlit as st
import time
from typing import IO, Any  # For type hinting uploaded files
from google.generativeai.types import FunctionDeclaration, Tool
from google.protobuf.json_format import MessageToDict
import google.generativeai as genai
import os
import fitz  # PyMuPDF
from proto.marshal.collections.maps import MapComposite
from proto.marshal.collections.repeated import RepeatedComposite

def extract_text(file_stream: IO[Any]) -> str:
    """
    MODIFIED: Extracts text from an in-memory file stream (from st.file_uploader).
    """
    text = ""
    # Read the bytes from the stream
    pdf_bytes = file_stream.read()
    
    # Open the PDF from bytes
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return "" # Return empty string on failure
    return text

# UNCHANGED: Your 'to_native' converter
def to_native(obj):
    """Recursively convert protobuf-like objects (MapComposite, RepeatedComposite) to native Python types."""
    if isinstance(obj, MapComposite):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, RepeatedComposite):
        return [to_native(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    else:
        return obj


def gem_json_job(job_text):
    """
    Extracts structured information from a Job Description using Gemini function calling.
    
    Args:
        job_text (str): The full text of the job description.
        model: The Gemini model instance (e.g., genai.GenerativeModel).
    
    Returns:
        dict: Extracted structured job details (title, company, requirements, etc.)
    """

    # Use your existing job extraction tool
    extract_job_details_func = FunctionDeclaration(
    name="extract_job_details",
    description="Extracts key details from a job description text.",
    parameters={
        "type": "object",
        "properties": {
            "Job_Title": {
                "type": "string",
                "description": "The official title of the job position."
            },
            "Company": {
                "type": "string",
                "description": "The company or organization offering the job."
            },
            "Location": {
                "type": "string",
                "description": "The city and/or country where the position is based."
            },
            "Responsibilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key duties and responsibilities expected from the candidate."
            },
            "Requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Essential technical and non-technical skills required for the job."
            },
            "Preferred_Qualifications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional or desired qualifications that give candidates an advantage."
            },
            "Duration": {
                "type": "string",
                "description": "The duration or contract type of the position (e.g., full-time, 3-month internship)."
            },
            "Start_Date": {
                "type": "string",
                "description": "The expected or mentioned start date of the position (if available)."
            },
            "Salary_or_Benefits": {
                "type": "string",
                "description": "Information about compensation or benefits, if specified."
            },
            "Application_Deadline": {
                "type": "string",
                "description": "The application deadline or closing date, if provided."
            },
            "Keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Important keywords that describe the role (skills, tools, topics)."
            },
            "Employment_Type": {
                "type": "string",
                "description": "The nature of employment (e.g., Internship, Full-time, Part-time, Contract)."
            },
        },
        "required": ["Job_Title", "Company", "Responsibilities", "Requirements"]
    }
)

    review_tool = Tool(function_declarations=[extract_job_details_func])

    # Create the prompt for Gemini
    extraction_prompt = f"""
    Please analyze the following Job Description and extract all relevant details such as:
    - Job Title
    - Company
    - Location
    - Responsibilities
    - Requirements
    - Preferred Qualifications
    - Duration (if internship)
    ---
    {job_text}
    ---
    """

    # Call Gemini API
    response = model.generate_content(
        extraction_prompt,
        tools=[review_tool],
        tool_config={"function_calling_config": "ANY"}
    )
    function_call_part = response.candidates[0].content.parts[0]
    function_call = function_call_part.function_call

    # Convert the MapComposite into a normal Python dict
    function_args = dict(function_call.args)
    native_data = to_native(function_args)


    # Safely access values
    extracted_data = {
        'Job_Title': native_data.get('Job_Title'),
        'Company': native_data.get('Company'),
        'Location': native_data.get('Location'),
        'Responsibilities': native_data.get('Responsibilities'),
        'Requirements': native_data.get('Requirements'),
        'Preferred_Qualifications': native_data.get('Preferred_Qualifications'),
        'Duration': native_data.get('Duration'),
    }

    # Convert to native Python types (if using protobuf types)

    return extracted_data
def gem_json(text):
# Create the prompt for the model
    extraction_prompt = f"""
    Please analyze the following CV and extract the required information.
    Here is the CV:
    ---
    {text}
    ---
    """

    # Make the API call, providing the tool and forcing the tool to be used
    extract_cv_details_func = FunctionDeclaration(
    name="extract_cv_details",
    description="Extracts key details from a CV text.",
    # This is now a dictionary, not a Schema object.
    parameters = {
    "type": "object",
    "properties": {
        "Name": {
            "type": "string",
            "description": "The applicant's full name"
        },
        "Contact_Info": {
            "type": "object",
            "properties": {
                "Email": {"type": "string"},
                "Phone": {"type": "string"},
                "LinkedIn": {"type": "string"},
                "Portfolio": {"type": "string"}
            }
        },
        "Education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Degree": {"type": "string"},
                    "Major": {"type": "string"},
                    "Institution": {"type": "string"},
                    "Graduation_Year": {"type": "string"},
                    "GPA": {"type": "string"}
                }
            }
        },
        "Experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Title": {"type": "string"},
                    "Company": {"type": "string"},
                    "Duration": {"type": "string"},
                    "Responsibilities": {"type": "string"},
                    "Technologies": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "Projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Title": {"type": "string"},
                    "Description": {"type": "string"},
                    "Technologies": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "Skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Technical and soft skills (e.g., Python, Machine Learning, Communication)"
        },
        "Certifications": {
            "type": "array",
            "items": {"type": "string"}
        },
        "Languages": {
            "type": "array",
            "items": {"type": "string"}
        },
        "Career_Objective": {
            "type": "string",
            "description": "Short statement about the applicant's professional goals"
        },
        "Soft_Skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Non-technical skills such as leadership, teamwork, or communication"
        },
        "Location": {
            "type": "string",
            "description": "Applicant's current city or country"
        },
        "Availability": {
            "type": "string",
            "description": "Whether the applicant is available full-time, part-time, or for internships"
        }
    },
    "required": ["Name", "Education", "Skills"]
}

)
    review_tool = Tool(function_declarations=[extract_cv_details_func])

    response = model.generate_content(
        extraction_prompt,
        tools=[review_tool],
        # By setting tool_config, we force the model to call our function
        tool_config={'function_calling_config': 'ANY'}
    )
    function_call_part = response.candidates[0].content.parts[0]
    function_call = function_call_part.function_call

    # Convert the MapComposite into a normal Python dict
    function_args = dict(function_call.args)
    native_data = to_native(function_args)
    # Now you can safely access values
    extracted_data = {
        'Name': native_data.get('Name'),
        'Education': native_data.get('Education'),
        'Skills': native_data.get('Skills'),
    }
    
    return extracted_data
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# # Let's check our models
# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)

model = genai.GenerativeModel('gemini-2.5-flash')


import json
import streamlit as st # You need streamlit for st.write/st.error
from typing import IO, Any # Add this if not already at the top

# (Make sure your robust gem_json and gem_json_job functions 
#  are defined above this)

def gem_evaluate(cv_file_stream: IO[Any], jd_input: Any):
    """
    This is the robust version of your function.
    It handles:
    1. Both text and file inputs for the Job Description.
    2. Errors from the upstream gem_json and gem_json_job functions.
    3. JSON parsing from the final model call.
    """
    
    # === Step 1: Process CV ===
    st.write("Step 1/3: Extracting data from CV...")
    cv_text = extract_text(cv_file_stream)
    if not cv_text:
        return {"error": "Failed to extract text from CV PDF."}

    # === Step 2: Process Job Description (flexible input) ===
    st.write("Step 2/3: Extracting data from Job Description...")
    des_text = ""
    if isinstance(jd_input, str):
        # It's a text string from st.text_area
        des_text = jd_input
        if not des_text.strip():
             return {"error": "Job Description text is empty."}
    else:
        # It's a file stream from st.file_uploader
        des_text = extract_text(jd_input)
        if not des_text:
            return {"error": "Failed to extract text from Job Description PDF."}
    
    # --- CHECKPOINT 1 ---
    # This assumes gem_json() returns a Python dictionary (or an error dict)
    cv_data = gem_json(cv_text)
    if "error" in cv_data:
        st.error("Stopping: Failed during CV data extraction.")
        return cv_data  # Stop and return the error
    # --- END CHECKPOINT ---

    # --- CHECKPOINT 2 ---
    # This assumes gem_json_job() returns a Python dictionary (or an error dict)
    des_data = gem_json_job(des_text)
    if "error" in des_data:
        st.error("Stopping: Failed during Job Description data extraction.")
        return des_data # Stop and return the error
    # --- END CHECKPOINT ---
    
    # --- Step 3: Final Evaluation ---
    # Convert the Python dicts to JSON strings for the prompt
    cv_json_str = json.dumps(cv_data, indent=2)
    des_json_str = json.dumps(des_data, indent=2)
    
    prompt = f"""
    You are an expert hiring manager. Evaluate how well the following CV matches the job description.
    
    Job Description (JSON):
    {des_json_str}
    
    Candidate CV (JSON):
    {cv_json_str}
    
    Please output a JSON with keys:
      - match_score (0-100)
      - present_skills (list)
      - missing_skills (list)
      - experience_diff (text)
      - suggestions (list)
    """
    
    st.write("Step 3/3: Evaluating match...")
    resp = model.generate_content(prompt)
    
    # This is your robust parsing logic, which is correct.
    try:
        raw_text = resp.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        result = json.loads(raw_text.strip())
        
    except (json.JSONDecodeError, AttributeError, TypeError):
        result = {"raw": resp.text, "error": "Failed to parse AI JSON response after cleaning."}
    except Exception as e:
        result = {"raw": str(resp), "error": f"An unexpected error occurred: {e}"}
            
    return result
# -------------------------------------------------------------------
# --- STREAMLIT APP UI ---
# -------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🤖 AI Job Application Evaluator")
st.write("Upload a candidate's CV (PDF) and a Job Description (PDF) to get an AI-powered match analysis.")

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader("1. Upload Candidate CV", type=["pdf"], help="Upload the candidate's resume in PDF format.")

with col2:
    jd_file = st.file_uploader("2. Upload Job Description", type=["pdf"], help="Upload the job requirements in PDF format.")

# --- Button and Logic ---
if st.button("Evaluate Match", type="primary"):
    
    # Check if both files are uploaded
    if cv_file is not None and jd_file is not None:
        
        # Show a spinner while processing
        with st.spinner("Analyzing... This may take a moment. 🤖"):
            # Pass the file objects directly to your function
            result = gem_evaluate(cv_file, jd_file)
        
        st.toast("Analysis complete!", icon="✅")
        
        # --- Display Results ---
        if "error" in result:
            st.error(f"An error occurred: {result['error']}")
            if "raw" in result:
                st.code(result['raw'], language="text")
        else:
            st.header("Analysis Results")
            
            # Show the score prominently
            score = result.get('match_score', 0)
            st.metric(label="Match Score", value=f"{score}%")
            st.progress(score)
            
            st.subheader("Key Insights")
            st.info(f"**Experience Fit:** {result.get('experience_diff', 'No analysis available.')}")
            
            st.subheader("💡 Suggestions for Interview")
            suggestions = result.get('suggestions', [])
            if suggestions:
                for sug in suggestions:
                    st.markdown(f"- {sug}")
            else:
                st.write("No specific suggestions provided.")
                
            # Use columns for the skills
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("✅ Skills Matched")
                present = result.get('present_skills', [])
                if present:
                    st.markdown("\n".join(f"- **{skill}**" for skill in present))
                else:
                    st.write("No matching skills identified.")

            with res_col2:
                st.subheader("❌ Skills Missing")
                missing = result.get('missing_skills', [])
                if missing:
                    st.markdown("\n".join(f"- {skill}" for skill in missing))
                else:
                    st.write("No missing skills identified.")

            # Show the raw JSON in an expander
            with st.expander("Show Raw JSON Response"):
                st.json(result)
                
    else:
        # Warning if one or both files are missing
        st.warning("Please upload *both* the CV and Job Description files.")