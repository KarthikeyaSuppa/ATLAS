import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import re
import requests
import json
from io import StringIO
import os
from groq import Groq
from dotenv import load_dotenv
import base64

# Load environment variables from .env file
load_dotenv()

# Access environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Page configuration and styling
st.set_page_config(layout="wide", page_title="AI Resume Screening System")

# Add custom CSS for styling
st.markdown("""
<style>
.highlighted {
    background-color: #FFD700;
    padding: 1px 3px;
    border-radius: 3px;
    font-weight: bold;
}

.modal {
    display: flex;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.7);
    justify-content: center;
    align-items: center;
}

.modal-content {
    background-color: #f9f9f9;
    margin: 5% auto;
    padding: 20px;
    border-radius: 10px;
    width: 90%;
    position: relative;
    max-height: 90vh;
    overflow-y: auto;
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #ddd;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

.modal-header h3 {
    margin: 0;
}

.close-button {
    font-size: 24px;
    font-weight: bold;
    cursor: pointer;
}

.modal-body {
    display: flex;
    gap: 20px;
}

.modal-section {
    flex: 1;
    padding: 15px;
    border-radius: 8px;
    background-color: #fff;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.modal-section h4 {
    margin-top: 0;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
}

.skill-score {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
}

.score-label {
    flex: 1;
}

.score-bar-container {
    flex: 2;
    background-color: #e0e0e0;
    height: 20px;
    border-radius: 10px;
    overflow: hidden;
}

.score-bar {
    height: 100%;
    background-color: #4CAF50;
    border-radius: 10px;
}

.score-value {
    margin-left: 10px;
    font-weight: bold;
}

/* Button styling */
.stButton>button {
    border-radius: 6px;
    font-weight: 500;
    padding: 0.25rem 1rem;
    background-color: #4CAF50;
    color: white;
}

.stButton>button:hover {
    background-color: #45a049;
    border-color: #45a049;
}

/* Center loading animation */
.loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    padding: 2rem;
}

/* Download button */
.download-btn {
    display: inline-flex;
    align-items: center;
    background-color: #2196F3;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    text-decoration: none;
    font-weight: 500;
    margin-top: 1rem;
}

.download-btn:hover {
    background-color: #0b7dda;
}

.download-icon {
    margin-right: 8px;
}

/* Custom header styling */
.app-header {
    background-color: #f8f9fa;
    padding: 1.5rem 1rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    text-align: center;
    border-bottom: 3px solid #4CAF50;
}

.app-header h1 {
    color: #333;
    margin-bottom: 0.5rem;
}

.app-header p {
    color: #666;
    font-size: 1.1rem;
}

/* Analysis button with icon */
.analyze-btn-container {
    display: flex;
    justify-content: center;
}

.analyze-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #4CAF50;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    font-weight: 500;
    width: 100%;
}

.analyze-btn:hover {
    background-color: #45a049;
}

.analyze-icon {
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# Text preprocessing function
def preprocess_text(text):
    """Clean text by converting to lowercase, removing punctuation, and normalizing whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    return text.strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Handle None values
    return text

# Highlight matched skills in resume
def highlight_matching_skills(resume_text, job_description):
    """Highlight text in resume that matches job description keywords."""
    keywords = extract_keywords(job_description)
    highlighted_text = resume_text
    
    # Create a pattern to match whole words only
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = r'\b' + re.escape(keyword) + r'\b'
        replacement = f'<span class="highlighted">{keyword}</span>'
        highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
    
    return highlighted_text

# Extract keywords from job description
def extract_keywords(job_description):
    """Extract important keywords from job description."""
    # Simple keyword extraction - could be enhanced with NLP libraries
    common_words = {'and', 'or', 'the', 'in', 'to', 'a', 'an', 'from', 'with', 'on', 'for', 'be', 'may', 'as', 'of'}
    words = job_description.lower().split()
    keywords = [word for word in words if word not in common_words and len(word) > 1]
    return set(keywords)

# NEW LLM-BASED RANKING FUNCTIONS

def rank_resumes_with_llm(job_description, resumes, groq_client):
    """
    Rank resumes using LLM analysis focused on skills, projects, and certifications.
    Returns scores and detailed analysis for each resume.
    """
    # Store results for each resume
    results = []
    
    # Step 1: Extract key requirements from job description using LLM
    jd_analysis = analyze_job_description(job_description, groq_client)
    
    # Step 2: Analyze each resume
    for resume_name, resume_text in resumes.items():
        # Get detailed analysis of resume content
        resume_analysis = analyze_resume_content(resume_text, jd_analysis, groq_client)
        
        # Calculate component scores
        skill_score = calculate_skill_match(resume_analysis.get("skills", []), jd_analysis.get("required_skills", []))
        project_score = calculate_project_relevance(resume_analysis.get("projects", []), jd_analysis.get("job_requirements", []))
        cert_score = calculate_certification_value(resume_analysis.get("certifications", []), jd_analysis.get("preferred_qualifications", []))
        
        # Calculate weighted final score with updated weights: skills 45%, projects 40%, certs 15%
        weighted_score = (skill_score * 0.45) + (project_score * 0.40) + (cert_score * 0.15)
        
        # Store results
        results.append({
            "resume_name": resume_name,
            "skill_score": skill_score,
            "project_score": project_score, 
            "certification_score": cert_score,
            "weighted_score": weighted_score,
            "analysis": resume_analysis
        })
    
    # Step 3: Normalize scores relative to other candidates
    max_score = max([r["weighted_score"] for r in results]) if results else 1
    for result in results:
        result["normalized_score"] = (result["weighted_score"] / max_score) * 10  # Scale to 0-10
    
    # Sort by normalized score
    sorted_results = sorted(results, key=lambda x: x["normalized_score"], reverse=True)
    
    return sorted_results

def analyze_job_description(job_description, groq_client):
    """
    Use LLM to extract key requirements from job description.
    """
    prompt = f"""
    Analyze the following job description and extract key requirements.
    Return a JSON object with the following structure:
    {{
        "required_skills": ["List of required technical and soft skills"],
        "job_requirements": ["List of core job requirements"],
        "preferred_qualifications": ["List of preferred qualifications"],
        "experience_level": "Required experience level"
    }}
    
    Ensure all fields are present and contain appropriate values.
    If a field is not found in the job description, provide an empty list or appropriate default value.
    
    JOB DESCRIPTION:
    {job_description}
    """
    
    messages = [
        {"role": "system", "content": "You are an expert HR analyst extracting key requirements from job descriptions. Always return a complete JSON object with all required fields."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Parse and validate the response
        analysis = json.loads(response.choices[0].message.content)
        
        # Ensure all required fields are present with default values if missing
        required_fields = {
            "required_skills": [],
            "job_requirements": [],
            "preferred_qualifications": [],
            "experience_level": "Not specified"
        }
        
        # Update with actual values from the analysis
        for field in required_fields:
            if field in analysis:
                required_fields[field] = analysis[field]
        
        return required_fields
        
    except Exception as e:
        st.error(f"Error analyzing job description: {str(e)}")
        # Return default structure in case of error
        return {
            "required_skills": [],
            "job_requirements": [],
            "preferred_qualifications": [],
            "experience_level": "Not specified"
        }

def analyze_resume_content(resume_text, job_requirements, groq_client):
    """
    Use LLM to analyze resume content with focus on skills, projects, and certifications.
    """
    # Create a prompt that includes the job requirements for context
    prompt = f"""
    Analyze the following resume in the context of the provided job requirements.
    Extract and evaluate:
    
    1. Skills found in the resume (both technical and soft skills)
    2. Projects completed with brief descriptions
    3. Certifications and educational qualifications
    4. Professional experience relevant to the job
    
    Job Requirements:
    - Required skills: {", ".join(job_requirements["required_skills"])}
    - Core job requirements: {job_requirements["job_requirements"]}
    - Preferred qualifications: {job_requirements["preferred_qualifications"]}
    
    RESUME:
    {resume_text}
    
    Return the analysis in JSON format with these categories and include relevance scores (0-10) for each skill, project, and certification found.
    """
    
    messages = [
        {"role": "system", "content": "You are an expert resume analyzer focusing on skills, projects, and certifications."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        # Parse and return the analysis
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        st.error(f"Error analyzing resume content: {str(e)}")
        # Return default structure in case of error
        return {
            "skills": [],
            "projects": [],
            "certifications": [],
            "experience": []
        }

def calculate_skill_match(candidate_skills, required_skills):
    """
    Calculate match score between candidate skills and required skills.
    Handles different formats of skills data.
    """
    # Handle different formats of candidate_skills
    all_candidate_skills = []
    
    # If it's a dictionary with technical/soft keys
    if isinstance(candidate_skills, dict):
        if "technical" in candidate_skills:
            all_candidate_skills.extend(candidate_skills["technical"])
        if "soft" in candidate_skills:
            all_candidate_skills.extend(candidate_skills["soft"])
    # If it's a list of skill objects
    elif isinstance(candidate_skills, list):
        all_candidate_skills = candidate_skills
    
    # Extract skill names based on dictionary format or list format
    if all_candidate_skills and isinstance(all_candidate_skills[0], dict):
        if "name" in all_candidate_skills[0]:
            candidate_skill_names = [skill["name"].lower() for skill in all_candidate_skills]
        else:
            # Handle if the skills are structured differently
            candidate_skill_names = []
            for skill in all_candidate_skills:
                # Get the first key if it exists
                keys = list(skill.keys())
                if keys:
                    candidate_skill_names.append(keys[0].lower())
    else:
        # If skills are just strings in a list
        candidate_skill_names = [str(skill).lower() for skill in all_candidate_skills]
    
    # Normalize required_skills to lowercase strings
    required_skill_names = [str(skill).lower() for skill in required_skills]
    
    if not required_skill_names:
        return 0.5  # Neutral score if no required skills
    
    # Count matches
    matches = 0
    skill_score_sum = 0
    
    for req_skill in required_skill_names:
        for i, cand_skill in enumerate(candidate_skill_names):
            # Check for match (including partial matches)
            if req_skill in cand_skill or cand_skill in req_skill:
                matches += 1
                
                # Get relevance score if available
                if isinstance(all_candidate_skills[i], dict) and "relevance" in all_candidate_skills[i]:
                    skill_score_sum += all_candidate_skills[i]["relevance"]
                elif isinstance(all_candidate_skills[i], dict) and "score" in all_candidate_skills[i]:
                    skill_score_sum += all_candidate_skills[i]["score"]
                elif isinstance(all_candidate_skills[i], dict) and "relevance_score" in all_candidate_skills[i]:
                    skill_score_sum += all_candidate_skills[i]["relevance_score"]
                else:
                    skill_score_sum += 5  # Default score
                break
    
    # Calculate coverage percentage
    coverage = matches / len(required_skill_names) if required_skill_names else 0
    
    # Calculate average skill relevance score
    avg_relevance = skill_score_sum / matches if matches > 0 else 0
    
    # Final skill score combines coverage and relevance
    return (coverage * 0.7) + ((avg_relevance / 10) * 0.3)

def calculate_project_relevance(projects, job_requirements):
    """
    Calculate relevance of candidate's projects to job requirements.
    """
    if not projects:
        return 0
    
    # Handle different formats of project data
    total_relevance = 0
    
    for project in projects:
        if isinstance(project, dict):
            # Try different possible keys for relevance score
            if "relevance_score" in project:
                total_relevance += project["relevance_score"]
            elif "relevance" in project:
                total_relevance += project["relevance"]
            elif "score" in project:
                total_relevance += project["score"]
            else:
                total_relevance += 5  # Default score
        else:
            total_relevance += 5  # Default score
    
    # Average the relevance scores of projects
    avg_relevance = total_relevance / len(projects)
    
    # Normalize to 0-1 scale
    return min(avg_relevance / 10, 1.0)

def calculate_certification_value(certifications, preferred_qualifications):
    """
    Calculate value of candidate's certifications to job.
    """
    if not certifications or not preferred_qualifications:
        return 0.5  # Neutral score if no certifications or preferences
    
    # Look for specific certification matches
    preferred_keywords = [str(qual).lower() for qual in preferred_qualifications]
    cert_score = 0
    
    for cert in certifications:
        # Handle certificate as either a dictionary or string
        if isinstance(cert, dict):
            if "name" in cert:
                cert_name = cert["name"].lower()
            else:
                # Get the first value if it's a dict without "name"
                values = list(cert.values())
                cert_name = str(values[0]).lower() if values else ""
        else:
            cert_name = str(cert).lower()
        
        # Check if certification matches any preferred qualification
        for keyword in preferred_keywords:
            if keyword in cert_name:
                # Get relevance score if available
                if isinstance(cert, dict):
                    if "relevance_score" in cert:
                        cert_score += cert["relevance_score"]
                    elif "relevance" in cert:
                        cert_score += cert["relevance"]
                    elif "score" in cert:
                        cert_score += cert["score"]
                    else:
                        cert_score += 5  # Default score
                else:
                    cert_score += 5  # Default score
                break
    
    # Normalize to 0-1 scale (cap at 10 for perfect score)
    max_possible = 10 * min(len(certifications), len(preferred_qualifications))
    normalized_score = min(cert_score / max_possible, 1) if max_possible > 0 else 0.5
    
    return normalized_score

# GROQ API integration for resume analysis
def analyze_resume_fit(resume_text, job_description):
    """
    Use Groq API with Llama-3.3-70B-Versatile to analyze how well the resume fits the job description.
    """
    try:
        # Initialize Groq client
        groq_client = Groq(api_key=groq_api_key)
        
        # Create prompt for analysis
        prompt = f"""
You are an expert HR analyst and recruitment specialist. Analyze how well the candidate's resume matches the job description.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Provide a detailed analysis in JSON format with the following structure:
{{
  "summary": "A comprehensive paragraph about how well the candidate fits the role",
  "strengths": ["List 3 specific strengths that align with the job requirements"],
  "areas_for_improvement": ["List 2 areas where the candidate could improve to better fit the role"],
  "fit_scores": {{
    "technical_expertise": 0.0,  // Score from 0-10
    "relevant_experience": 0.0,  // Score from 0-10
    "educational_background": 0.0  // Score from 0-10
  }}
}}

Ensure the scores are realistic evaluations from 0-10 based on how well the resume matches the specific requirements in the job description.
        """
        
        # Setup request
        messages = [
            {"role": "system", "content": "You are an expert HR analyst providing candidate evaluations in JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        # Make API request
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        analysis_text = response.choices[0].message.content
        analysis = json.loads(analysis_text)
        
        return analysis
        
    except Exception as e:
        st.error(f"Analysis Error: {str(e)}")
        # Fallback analysis in case of API failure
        return {
            "summary": "Unable to analyze resume due to API error. Please try again later.",
            "strengths": ["API connection failed"],
            "areas_for_improvement": ["Check API connection"],
            "fit_scores": {
                "technical_expertise": 5.0,
                "relevant_experience": 5.0,
                "educational_background": 5.0
            }
        }

# Function to create downloadable PDF report
def create_downloadable_report(resume_name, analysis_results, highlighted_text, job_description):
    """Create a downloadable HTML report of the analysis."""
    html_content = f"""
    <html>
    <head>
        <title>Resume Analysis Report: {resume_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            h1 {{ color: #2C3E50; text-align: center; }}
            h2 {{ color: #2980B9; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            .section {{ margin-bottom: 30px; background: #f9f9f9; padding: 20px; border-radius: 5px; }}
            .highlighted {{ background-color: #FFD700; padding: 1px 3px; border-radius: 3px; }}
            .score-container {{ display: flex; align-items: center; margin-bottom: 15px; }}
            .score-label {{ flex: 1; }}
            .score-bar-container {{ flex: 2; background-color: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden; }}
            .score-bar {{ height: 100%; background-color: #4CAF50; border-radius: 10px; }}
            .score-value {{ margin-left: 10px; font-weight: bold; }}
            .summary {{ padding: 10px; background: #EFF8FB; border-left: 3px solid #3498DB; margin-bottom: 20px; }}
            .strengths {{ color: #27AE60; }}
            .improvements {{ color: #E74C3C; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Resume Analysis Report</h1>
            <div class="section">
                <h2>Resume: {resume_name}</h2>
                <h3>Job Description Summary</h3>
                <p>{job_description[:300]}...</p>
            </div>
            
            <div class="section">
                <h2>Fit Analysis</h2>
                <div class="summary">
                    <p><strong>Summary:</strong> {analysis_results["summary"]}</p>
                </div>
                
                <h3 class="strengths">Key Strengths:</h3>
                <ul>
    """
    
    # Add strengths
    for strength in analysis_results["strengths"]:
        html_content += f"<li>{strength}</li>"
    
    html_content += """
                </ul>
                
                <h3 class="improvements">Areas for Improvement:</h3>
                <ul>
    """
    
    # Add areas for improvement
    for area in analysis_results["areas_for_improvement"]:
        html_content += f"<li>{area}</li>"
    
    html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Fit Scores</h2>
    """
    
    # Add score bars
    for category, score in analysis_results["fit_scores"].items():
        category_name = " ".join(word.capitalize() for word in category.split("_"))
        percentage = score * 10
        html_content += f"""
                <div class="score-container">
                    <div class="score-label">{category_name}</div>
                    <div class="score-bar-container">
                        <div class="score-bar" style="width: {percentage}%;"></div>
                    </div>
                    <div class="score-value">{score}/10</div>
                </div>
        """
    
    html_content += """
            </div>
            
            <div class="section">
                <h2>Skills Matched</h2>
    """
    
    html_content += highlighted_text
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Encode as base64 for download
    b64 = base64.b64encode(html_content.encode()).decode()
    
    return b64
    
# Generate download link
def get_download_link(b64, filename):
    """Generate a download link for the report."""
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" class="download-btn"><span class="download-icon">📥</span> Download Analysis Report</a>'
    return href

# Initialize session state variables
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'resume_texts' not in st.session_state:
    st.session_state.resume_texts = {}
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'current_resume' not in st.session_state:
    st.session_state.current_resume = ""
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'highlighted_text' not in st.session_state:
    st.session_state.highlighted_text = ""
if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = ""
if 'ranked_results' not in st.session_state:
    st.session_state.ranked_results = []
if 'job_analysis' not in st.session_state:
    st.session_state.job_analysis = {}

# App Header
st.markdown("""
<div class="app-header">
    <h1>AI Resume Screening & Candidate Ranking System</h1>
    <p>Powered by LLM Analysis for intelligent resume matching and evaluation</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for input and results
tab1, tab2 = st.tabs(["📝 Input Data", "📊 Results"])

with tab1:
    # Job Description input
    st.header("Job Description")
    job_description_input = st.text_area("Enter the job description", height=200, placeholder="Paste the full job description here...")
    
    # Resume upload
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    # Process button with improved styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Process Resumes", use_container_width=True):
            if not job_description_input or not uploaded_files:
                st.error("Please provide both job description and upload at least one resume.")
            else:
                st.session_state.job_description = job_description_input
                st.session_state.uploaded_files = uploaded_files
                
                with st.spinner('Processing resumes with AI analysis...'):
                    processed_files = []
                    resume_texts = {}
                    
                    # Process each uploaded resume
                    for file in uploaded_files:
                        try:
                            text = extract_text_from_pdf(file)
                            if text.strip():
                                processed_files.append(file)
                                resume_texts[file.name] = text  # Store original text for display
                            else:
                                st.warning(f"No text extracted from {file.name}. It might be a scanned image.")
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                    
                    # Check if any resumes were successfully processed
                    if not resume_texts:
                        st.error("No valid resumes were processed. Please check your files.")
                    else:
                        try:
                            # Initialize Groq client
                            groq_client = Groq(api_key=groq_api_key)
                            
                            # Analyze job description first
                            with st.status("Analyzing job description..."):
                                job_analysis = analyze_job_description(job_description_input, groq_client)
                                st.session_state.job_analysis = job_analysis
                            
                            # Analyze and rank resumes
                            with st.status("Ranking resumes..."):
                                ranked_results = rank_resumes_with_llm(job_description_input, 
                                                                     resume_texts, 
                                                                     groq_client)
                            
                            # Store results in session state
                            st.session_state.resume_texts = resume_texts
                            st.session_state.ranked_results = ranked_results
                            st.session_state.processed_files = processed_files
                            st.session_state.processing_done = True
                            
                            # Switch to results tab
                            st.success("Processing complete! View results in the Results tab.")
                        except Exception as e:
                            st.error(f"Error during analysis process: {str(e)}")
                
with tab2:
    if st.session_state.processing_done:
        # Display job analysis
        st.header("Job Analysis")
        with st.expander("View Job Requirements", expanded=True):
            st.subheader("Required Skills")
            for skill in st.session_state.job_analysis.get("required_skills", []):
                st.write(f"- {skill}")
                
            st.subheader("Core Job Requirements")
            for req in st.session_state.job_analysis.get("job_requirements", []):
                st.write(f"- {req}")
                
            st.subheader("Preferred Qualifications")
            for qual in st.session_state.job_analysis.get("preferred_qualifications", []):
                st.write(f"- {qual}")
            
            st.write(f"**Experience Level:** {st.session_state.job_analysis.get('experience_level', 'Not specified')}")
        
        # Display ranked candidates
        st.header("Ranked Candidates")
        
        # Check if we have results
        if not st.session_state.ranked_results:
            st.warning("No ranking results available. Please process resumes first.")
        else:
            # Create a dataframe for the rankings
            ranking_data = []
            for result in st.session_state.ranked_results:
                ranking_data.append({
                    "Resume": result["resume_name"],
                    "Overall Score": f"{result['normalized_score']:.1f}/10",
                    "Skills Match": f"{result['skill_score']*100:.1f}%",
                    "Project Relevance": f"{result['project_score']*100:.1f}%",
                    "Certification Value": f"{result['certification_score']*100:.1f}%"
                })
            
            ranking_df = pd.DataFrame(ranking_data)
            st.dataframe(ranking_df, use_container_width=True)
            
            # Add options to view individual resume details
            st.subheader("View Detailed Resume Analysis")
            selected_resume = st.selectbox("Select a resume to view detailed analysis:", 
                                        [result["resume_name"] for result in st.session_state.ranked_results])
            
            if selected_resume:
                st.session_state.current_resume = selected_resume
                
                # Find the corresponding result
                selected_result = next((res for res in st.session_state.ranked_results if res["resume_name"] == selected_resume), None)
                
                if selected_result:
                    # Analyze the selected resume in detail
                    with st.spinner("Generating detailed analysis..."):
                        # Initialize Groq client
                        groq_client = Groq(api_key=groq_api_key)
                        
                        # Get detailed analysis
                        analysis_results = analyze_resume_fit(
                            st.session_state.resume_texts[selected_resume],
                            st.session_state.job_description
                        )
                        
                        # Highlight matching skills
                        highlighted_text = highlight_matching_skills(
                            st.session_state.resume_texts[selected_resume],
                            st.session_state.job_description
                        )
                        
                        st.session_state.analysis_results = analysis_results
                        st.session_state.highlighted_text = highlighted_text
                        st.session_state.show_analysis = True
                        
                        # Generate downloadable report
                        report_b64 = create_downloadable_report(
                            selected_resume,
                            analysis_results,
                            highlighted_text,
                            st.session_state.job_description
                        )
                        st.session_state.analysis_report = report_b64
                
                # Display detailed analysis if available
                if st.session_state.show_analysis:
                    # Analysis Results
                    st.subheader("Resume Analysis")
                    
                    # Summary
                    with st.container():
                        st.markdown("### Fit Summary")
                        st.info(st.session_state.analysis_results["summary"])
                    
                    # Two columns for strengths and improvements
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Key Strengths")
                        for strength in st.session_state.analysis_results["strengths"]:
                            st.markdown(f"✅ {strength}")
                    
                    with col2:
                        st.markdown("### Areas for Improvement")
                        for area in st.session_state.analysis_results["areas_for_improvement"]:
                            st.markdown(f"🔍 {area}")
                    
                    # Fit Scores
                    st.markdown("### Fit Scores")
                    for category, score in st.session_state.analysis_results["fit_scores"].items():
                        category_name = " ".join(word.capitalize() for word in category.split("_"))
                        st.markdown(f"**{category_name}:** {score}/10")
                        st.progress(score/10)
                    
                    # Resume with Highlighted Skills
                    with st.expander("Resume with Highlighted Skills", expanded=False):
                        st.markdown(st.session_state.highlighted_text, unsafe_allow_html=True)
                    
                    # Download report button
                    if st.session_state.analysis_report:
                        filename = f"Resume_Analysis_{selected_resume.split('.')[0]}.html"
                        st.markdown(get_download_link(st.session_state.analysis_report, filename), unsafe_allow_html=True)
    else:
        st.info("No resumes processed yet. Please upload resumes and job description in the Input Data tab.")

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #777;">
    <p>AI Resume Screening System | Powered by LLM Analysis</p>
</div>
""", unsafe_allow_html=True)