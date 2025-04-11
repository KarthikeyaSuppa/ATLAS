import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
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

# Train FastText model
def train_fasttext(resumes):
    """Train a FastText model on the provided texts."""
    sentences = [text.split() for text in resumes]
    model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
    return model

# Rank resumes with hybrid scoring
def rank_resumes(job_description, resumes):
    """Rank resumes using a hybrid score of cosine similarity and keyword matching."""
    # Train FastText model on job description and resumes
    model = train_fasttext([job_description] + resumes)

    # Helper function to get average word vector
    def get_vector(text):
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    # Compute cosine similarity
    job_vector = get_vector(job_description)
    resume_vectors = [get_vector(resume) for resume in resumes]
    cosine_scores = cosine_similarity([job_vector], resume_vectors).flatten()

    # Compute keyword matching score
    jd_words = set(job_description.split())
    keyword_scores = []
    for resume in resumes:
        resume_words = set(resume.split())
        score = len(jd_words.intersection(resume_words)) / len(jd_words) if jd_words else 0
        keyword_scores.append(score)

    # Combine scores with equal weights
    hybrid_scores = 0.5 * cosine_scores + 0.5 * np.array(keyword_scores)
    return hybrid_scores

# Extract keywords from job description
def extract_keywords(job_description):
    """Extract important keywords from job description."""
    # Simple keyword extraction - could be enhanced with NLP libraries
    common_words = {'and', 'or', 'the', 'in', 'to', 'a', 'an', 'from', 'with', 'on', 'for', 'be', 'may', 'as', 'of'}
    words = job_description.lower().split()
    keywords = [word for word in words if word not in common_words and len(word) > 2]
    return set(keywords)

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

# GROQ API integration for LLM analysis
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
            model="llama-3.1-8b-instant",
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
if 'scores' not in st.session_state:
    st.session_state.scores = []
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

# App Header
st.markdown("""
<div class="app-header">
    <h1>AI Resume Screening & Candidate Ranking System</h1>
    <p>Powered by FastText and Llama-3.1-8b-instant for intelligent resume matching and analysis</p>
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
                
                with st.spinner('Processing resumes and computing scores...'):
                    processed_files = []
                    resumes = []
                    resume_texts = {}
                    
                    # Process each uploaded resume
                    for file in uploaded_files:
                        try:
                            text = extract_text_from_pdf(file)
                            if text.strip():
                                processed_files.append(file)
                                processed_text = preprocess_text(text)
                                resumes.append(processed_text)
                                resume_texts[file.name] = text  # Store original text for display
                            else:
                                st.warning(f"No text extracted from {file.name}. It might be a scanned image.")
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                    
                    # Check if any resumes were successfully processed
                    if not resumes:
                        st.error("No valid resumes were processed. Please check your files.")
                    else:
                        # Preprocess job description and compute scores
                        preprocessed_jd = preprocess_text(job_description_input)
                        scores = rank_resumes(preprocessed_jd, resumes)
                        
                        # Store results in session state
                        st.session_state.resume_texts = resume_texts
                        st.session_state.scores = scores
                        st.session_state.processed_files = processed_files
                        st.session_state.processing_done = True
                        
                        # Switch to results tab
                        st.success("Processing complete! View results in the Results tab.")

with tab2:
    if st.session_state.processing_done:
        # Create a three-column layout
        left_col, middle_col, right_col = st.columns([1, 1, 1])
        
        with left_col:
            st.subheader("Job Description")
            st.text_area("Job Description Text", st.session_state.job_description, height=300, disabled=True)
        
        with middle_col:
            st.subheader("Resume Rankings")
            # Create and display results
            results = pd.DataFrame({
                "Resume": [file.name for file in st.session_state.processed_files],
                "Score": st.session_state.scores
            })
            results = results.sort_values(by="Score", ascending=False)
            results["Score"] = results["Score"].apply(lambda x: f"{x:.2f}")
            st.dataframe(results, use_container_width=True)
        
        with right_col:
            st.subheader("Resume Content")
            for idx, (file_name, text) in enumerate(st.session_state.resume_texts.items()):
                with st.expander(f"{file_name}"):
                    st.text_area(f"Resume Content", text, height=200, disabled=True, key=f"resume_text_{idx}")
                    
                    # Improved analyze button with icon
                    st.markdown(f"""
                    <div class="analyze-btn-container">
                        <button id="analyze_btn_{idx}" class="analyze-btn" onclick="document.getElementById('analyze_button_{idx}').click()">
                            <span class="analyze-icon">🔍</span>
                            Analyze Resume
                        </button>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hidden button for the JavaScript to click
                    if st.button("Analyze", key=f"analyze_button_{idx}", help="Analyze skills and fit for job"):
                        st.write("Analyze button clicked")  # Debugging statement

                        # Create a placeholder for loading animation
                        analysis_placeholder = st.empty()
                        with analysis_placeholder.container():
                            st.markdown("""
                            <div class="loading-container">
                                <h3>Analyzing resume with AI...</h3>
                                <img src="https://www.icegif.com/wp-content/uploads/loading-icegif-1.gif" width="150" />
                                <p>This may take a few moments...</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            try:
                                # Perform the analysis
                                st.session_state.show_analysis = True
                                st.session_state.current_resume = file_name
                                st.session_state.highlighted_text = highlight_matching_skills(text, st.session_state.job_description)
                                st.session_state.analysis_results = analyze_resume_fit(text, st.session_state.job_description)
                                
                                st.write("Analysis completed successfully")  # Debugging statement
                                
                                # Create downloadable report
                                b64_report = create_downloadable_report(
                                    file_name, 
                                    st.session_state.analysis_results,
                                    st.session_state.highlighted_text,
                                    st.session_state.job_description
                                )
                                st.session_state.analysis_report = b64_report
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                                st.write(f"Exception occurred: {str(e)}")  # Debugging statement
                            
                        # Clear the loading animation
                        analysis_placeholder.empty()
                        
                        # Use st.rerun() to refresh the page
                        st.rerun()

        # Display analysis results below the existing content
        if st.session_state.show_analysis:
            st.subheader(f"Detailed Analysis: {st.session_state.current_resume}")
            
            # Download button
            st.markdown(get_download_link(
                st.session_state.analysis_report, 
                f"Resume_Analysis_{st.session_state.current_resume.replace(' ', '_')}.html"
            ), unsafe_allow_html=True)
            
            # Analysis sections
            st.markdown("<h4>🔍 Skills Matched</h4>", unsafe_allow_html=True)
            st.markdown(st.session_state.highlighted_text, unsafe_allow_html=True)
            
            st.markdown("<h4>📋 Fit Analysis</h4>", unsafe_allow_html=True)
            st.write(st.session_state.analysis_results["summary"])
            
            st.markdown("<h4>Key Strengths:</h4>", unsafe_allow_html=True)
            for strength in st.session_state.analysis_results["strengths"]:
                st.markdown(f"- {strength}", unsafe_allow_html=True)
                
            st.markdown("<h4>Areas for Improvement:</h4>", unsafe_allow_html=True)
            for area in st.session_state.analysis_results["areas_for_improvement"]:
                st.markdown(f"- {area}", unsafe_allow_html=True)
                
            st.markdown("<h4>📊 Fit Score (out of 10)</h4>", unsafe_allow_html=True)
            for category, score in st.session_state.analysis_results["fit_scores"].items():
                category_name = " ".join(word.capitalize() for word in category.split("_"))
                percentage = score * 10
                st.markdown(f"""
                <div class="skill-score">
                    <div class="score-label">{category_name}</div>
                    <div class="score-bar-container">
                        <div class="score-bar" style="width: {percentage}%;"></div>
                    </div>
                    <div class="score-value">{score}/10</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Please upload resumes and job description in the Input Data tab.")