# AI Job Application Evaluator (Streamlit + Gemini)

A Streamlit application that evaluates how well a candidate’s resume matches a job description. The app extracts text from uploaded PDFs, uses the Gemini API to produce structured CV/JD summaries, then returns a final evaluation including a match score, matched skills, missing skills, experience fit, and actionable suggestions.

## Live Demo

Streamlit App: [Resume Evaluator](https://resume1evaluator.streamlit.app/)

## Key Features

- **PDF resume + job description ingestion** using PyMuPDF (no OCR required for text-based PDFs)
- **LLM-first extraction pipeline** (Gemini) that converts unstructured resume and JD text into clean structured JSON
- **Scoring + gap analysis** returning:
  - match score (0–100)
  - present skills
  - missing required skills
  - experience comparison summary
  - concrete improvement suggestions
- **Robust response handling** with retry support and JSON-cleaning logic to reduce formatting failures
- **User-friendly Streamlit UI** with progress steps, score metric, progress bar, and raw JSON view

## How It Works

1. **Upload PDFs**: Candidate CV and Job Description
2. **Text Extraction**: Extracts raw text from PDFs using PyMuPDF
3. **LLM Structuring**:
   - CV → structured JSON (skills, experience, education, certifications, projects, summary)
   - JD → structured JSON (title, company, required skills, preferred skills, responsibilities, experience level)
4. **Final Evaluation**:
   - Gemini generates a single JSON output with score and skill gaps based on a weighted scoring rubric

## Output Format

The application returns a JSON object with:

- `match_score` (0–100)
- `present_skills` (list)
- `missing_skills` (list)
- `experience_diff` (string)
- `suggestions` (list)

## Project Structure

```text
.
├── app.py / main.py                  # Streamlit app entry point (UI + pipeline)
└── requirements.txt  
