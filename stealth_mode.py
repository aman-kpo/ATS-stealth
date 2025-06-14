import streamlit as st
import json
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import pdfplumber
import PyPDF2
import fitz 
from PIL import Image
import easyocr
import os
import tempfile
import re
import io

# Fix for Windows OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set USER_AGENT to avoid warnings
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Set page configuration
st.set_page_config(
    page_title="ATS_Assassin",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    
    /* Enhanced score card styling */
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .score-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    
    .score-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .score-card-initial {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .score-card-improved {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .score-card-improvement {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .score-label {
        color: white;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .score-value {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .score-delta {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.3rem;
    }
    
    .arrow-up {
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-bottom: 12px solid #4ade80;
        display: inline-block;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Progress ring for scores */
    .progress-ring {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto 1rem auto;
    }
    
    .progress-ring-circle {
        stroke: rgba(255,255,255,0.3);
        fill: transparent;
        stroke-width: 8;
    }
    
    .progress-ring-circle-progress {
        stroke: white;
        fill: transparent;
        stroke-width: 8;
        stroke-linecap: round;
        transform: rotate(-90deg);
        transform-origin: 50% 50%;
        transition: stroke-dashoffset 1s ease-in-out;
    }
    
    .section-header {
        background-color: #e9ecef;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    
    .section-content {
        padding: 1rem;
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    /* Animated gradient background for results header */
    .results-header {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .results-header h1 {
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Metric styles for native Streamlit metrics */
    [data-testid="metric-container"] {
        background: transparent;
        padding: 0;
    }
    
    [data-testid="metric-container"] > div {
        background: transparent;
    }
    
    [data-testid="metric-container"] label {
        color: white !important;
        font-weight: 500;
        opacity: 0.9;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: white !important;
        font-size: 2.2rem;
        font-weight: bold;
    }
    
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: white !important;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Define Pydantic models
class ResumeSection(BaseModel):
    """Information about a section in a resume."""
    section_name: str = Field(description="The name of the resume section (e.g., 'Experience', 'Education', 'Skills').")
    content: str = Field(description="The full text content of this section.")

class ResumeAnalysis(BaseModel):
    """Analysis of a resume based on a job description."""
    resume_sections: list[ResumeSection] = Field(description="A list of identified sections and their content from the resume.")
    initial_match_score: int = Field(description="An initial match score out of 100, indicating how well the resume matches the job description.")
    score_issues: str = Field(description="A detailed summary of the main reasons why the initial score is not 100, specifically identifying key requirements from the job description that are missing or not clearly highlighted in the resume.")
    suggested_resume_updates: str = Field(description="Specific, actionable suggestions for how to modify the resume content to better match the job description. Focus on adding relevant keywords and rephrasing existing content to align with job requirements.")
    updated_resume_content_suggestion: list[ResumeSection] = Field(description="A suggested revised version of the resume content, presented as a list of sections with potentially modified content based on the job description. These modifications should be realistic and significantly improve the match score.")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'ocr_reader' not in st.session_state:
    st.session_state.ocr_reader = None

# Helper functions
@st.cache_resource
def get_ocr_reader():
    """Initialize OCR reader once and cache it"""
    try:
        return easyocr.Reader(['en'], gpu=False)  # Disable GPU for Windows compatibility
    except Exception as e:
        st.warning(f"OCR initialization warning: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using multiple methods without requiring poppler."""
    
    # Method 1: Try pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if pdf.pages:
                all_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n"

                if all_text.strip():
                    return all_text.strip()
                else:
                    st.info("pdfplumber did not extract text. Trying PyPDF2...")
    except Exception as e:
        st.warning(f"pdfplumber failed: {e}. Trying PyPDF2...")
    
    # Method 2: Try PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            all_text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
            
            if all_text.strip():
                return all_text.strip()
            else:
                st.info("PyPDF2 did not extract text. Trying PyMuPDF with OCR...")
    except Exception as e:
        st.warning(f"PyPDF2 failed: {e}. Trying PyMuPDF with OCR...")
    
    # Method 3: Use PyMuPDF (fitz) to convert PDF to images for OCR
    try:
        # Open the PDF with PyMuPDF
        pdf_document = fitz.open(pdf_path)
        
        if len(pdf_document) == 0:
            return "The PDF file is empty or has no pages."
        
        # Get the first page
        page = pdf_document[0]
        
        # Convert page to image
        mat = fitz.Matrix(2, 2)  # Increase resolution
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(img_data))
        
        # Save the image temporarily for OCR
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
            image_path = tmp_img.name
            image.save(image_path, 'PNG')
        
        # Get or initialize OCR reader
        reader = get_ocr_reader()
        if reader is None:
            return "OCR reader could not be initialized."
        
        # Read text from the image
        result = reader.readtext(image_path)
        
        # Extract text
        extracted_text = ""
        for (bbox, text, prob) in result:
            extracted_text += text + " "
        
        # Clean up
        pdf_document.close()
        try:
            os.unlink(image_path)
        except:
            pass
        
        if extracted_text.strip():
            return extracted_text.strip()
        else:
            return "Could not extract text from the PDF using any method."
            
    except Exception as e:
        return f"All text extraction methods failed: {e}"

def analyze_and_update_resume(resume_text: str, job_description_text: str):
    """Analyze resume against job description and provide suggestions."""
    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert ATS scorer and resume optimization assistant.
            Your primary goal is to provide actionable and realistic suggestions to *significantly improve* a resume's match score against a given job description.
            Your task is to analyze a resume against a job description with high accuracy, simulating an advanced ATS system.
            First, meticulously identify ALL key sections and their content from the provided resume text.
            Then, perform a detailed assessment of how well the resume's content, skills, and experience align with the requirements and preferences outlined in the job description. Provide an initial match score out of 100, explaining your reasoning based on specific keywords, skills, and experiences mentioned in both the resume and job description.
            Clearly articulate the specific gaps or areas where the resume significantly deviates from or fails to address key aspects of the job description. These should be concrete points directly tied to the job requirements.
            Based on this analysis, provide specific and impactful suggestions for modifying the resume content. **These suggestions must be designed to strategically increase the resume's ATS score against the job description by directly addressing the identified gaps and highlighting relevant experience.** Focus on:
            1. Incorporating relevant keywords and phrases from the job description where applicable and where supported by the candidate's experience.
            2. Rephrasing existing bullet points or descriptions to *strongly* highlight experiences that are directly relevant to the job requirements and use action verbs that match the job description's tone.
            3. Adding missing information if it's implied by the existing content but not explicitly stated (e.g., mentioning specific tools, methodologies, or quantifiable results used) that are mentioned in the job description.
            4. **Ensure the suggested modifications realistically improve the alignment and are not fabricated.**
            Finally, present the suggested revised version of the resume content. This should be a realistic and optimized version of the original resume, incorporating the suggested changes within the original section structure. Do NOT invent experience or skills that are not at least partially supported by the original resume content; focus on strategically highlighting and re-framing existing information to better match the job description and improve the score.
            Ensure the output is strictly in the specified JSON format. The 'updated_resume_content_suggestion' field should contain the full text of the suggested updated resume sections."""),
            ("human", """Resume Text:
            {resume_text}

            Job Description Text:
            {job_description_text}"""),
        ])

        structured_llm = llm.with_structured_output(ResumeAnalysis)
        analysis_chain = prompt | structured_llm

        # Perform initial analysis
        with st.spinner("Analyzing resume..."):
            initial_analysis = analysis_chain.invoke({
                "resume_text": resume_text,
                "job_description_text": job_description_text
            })

        # Prepare updated resume text
        suggested_updated_resume_text_sections = ""
        for section in initial_analysis.updated_resume_content_suggestion:
            suggested_updated_resume_text_sections += f"{section.section_name}:\n{section.content}\n\n"

        # Score the updated resume
        with st.spinner("Calculating improved score..."):
            scoring_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert resume and job description matcher.
                Your task is to assess how well the provided resume text matches the job description and provide a single integer score out of 100 based on ATS scoring principles. Focus on the alignment of skills, experience, and keywords. Provide only the integer score."""),
                ("human", """Assess the match score between the following resume text and job description.
                Provide only a single integer score out of 100.

                Resume Text:
                {resume_text}

                Job Description Text:
                {job_description_text}"""),
            ])

            scoring_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            scoring_chain = scoring_prompt | scoring_llm

            updated_score_response = scoring_chain.invoke({
                "resume_text": suggested_updated_resume_text_sections.strip(),
                "job_description_text": job_description_text
            })

            # Parse score
            try:
                updated_score_text = updated_score_response.content.strip()
                match = re.search(r'\d+', updated_score_text)
                if match:
                    updated_score = int(match.group(0))
                    updated_score = max(0, min(100, updated_score))
                else:
                    updated_score = -1
            except:
                updated_score = -1

        # Prepare result - using model_dump() instead of dict() for Pydantic v2
        result = {
            "initial_analysis": initial_analysis.model_dump(),
            "initial_score": initial_analysis.initial_match_score,
            "suggested_resume_updates_description": initial_analysis.suggested_resume_updates,
            "suggested_updated_resume_content_json": [sec.model_dump() for sec in initial_analysis.updated_resume_content_suggestion],
            "updated_score": updated_score,
            "score_comparison": f"Initial score: {initial_analysis.initial_match_score}, Updated score: {updated_score}"
        }

        return result

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        return {"error": str(e)}

# Main app
st.title("🎯 ATS Assassin - Stealth Mode")
st.markdown("Upload your resume and job description to get AI-powered suggestions for improvement to enhance the ATS score.")

# Installation instructions
with st.expander("📦 Required Dependencies", expanded=False):
    st.markdown("""
    **Install the following packages:**
    ```bash
    pip install streamlit langchain-core langchain-openai langchain-community
    pip install pydantic pdfplumber PyPDF2 pymupdf pillow easyocr
    ```
    
    **Note:** This version doesn't require poppler! We use PyMuPDF (fitz) instead of pdf2image.
    """)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI API Key
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    
    # Job Description Input Method
    st.subheader("📋 Job Description Input")
    input_method = st.radio(
        "Choose input method:",
        ["Upload PDF", "Enter URL"],
        help="Select how you want to provide the job description"
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Resume")
    resume_file = st.file_uploader("Choose your resume PDF", type=['pdf'])
    
    if resume_file:
        st.success(f"✅ Resume uploaded: {resume_file.name}")

with col2:
    st.subheader("💼 Job Description")
    
    job_description_text = None
    
    if input_method == "Upload PDF":
        job_file = st.file_uploader("Choose job description PDF", type=['pdf'])
        if job_file:
            st.success(f"✅ Job description uploaded: {job_file.name}")
    else:
        job_url = st.text_input("Enter job description URL", placeholder="https://example.com/job-posting")
        if job_url:
            st.success(f"✅ URL provided")

# Analyze button
if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
    # Validation
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not resume_file:
        st.error("Please upload your resume.")
    elif input_method == "Upload PDF" and not job_file:
        st.error("Please upload the job description PDF.")
    elif input_method == "Enter URL" and not job_url:
        st.error("Please enter the job description URL.")
    else:
        try:
            # Extract resume text
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_resume:
                tmp_resume.write(resume_file.read())
                tmp_resume.flush()
                resume_text = extract_text_from_pdf(tmp_resume.name)
                try:
                    os.unlink(tmp_resume.name)
                except:
                    pass
            
            # Extract job description
            if input_method == "Upload PDF":
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_job:
                    tmp_job.write(job_file.read())
                    tmp_job.flush()
                    job_description_text = extract_text_from_pdf(tmp_job.name)
                    try:
                        os.unlink(tmp_job.name)
                    except:
                        pass
            else:
                with st.spinner("Fetching job description from URL..."):
                    loader = WebBaseLoader(job_url)
                    docs = loader.load()
                    # Try to get description from metadata, fallback to page content
                    if docs and len(docs) > 0:
                        job_description_text = docs[0].metadata.get("description", docs[0].page_content)
                    else:
                        st.error("Could not fetch content from the URL")
                        job_description_text = None
            
            if job_description_text:
                # Perform analysis
                result = analyze_and_update_resume(resume_text, job_description_text)
                
                if "error" not in result:
                    st.session_state.analysis_complete = True
                    st.session_state.analysis_result = result
                else:
                    st.error(f"Analysis failed: {result['error']}")
            else:
                st.error("Could not extract job description text")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

# Display results
if st.session_state.analysis_complete and st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    # Animated header
    st.markdown('<div class="results-header"><h1>📊 Analysis Results</h1></div>', unsafe_allow_html=True)
    
    # Score comparison with enhanced styling
    col1, col2, col3 = st.columns([1, 1, 1])
    
    initial_score = result['initial_score']
    updated_score = result['updated_score']
    score_diff = updated_score - initial_score if updated_score != -1 else 0
    
    with col1:
        st.markdown('<div class="score-card score-card-initial">', unsafe_allow_html=True)
        # Progress ring SVG
        progress_initial = initial_score / 100 * 377  # 377 is the circumference of the circle
        st.markdown(f'''
        <div class="progress-ring">
            <svg width="120" height="120">
                <circle class="progress-ring-circle" cx="60" cy="60" r="50"></circle>
                <circle class="progress-ring-circle-progress" cx="60" cy="60" r="50" 
                    style="stroke-dasharray: 377; stroke-dashoffset: {377 - progress_initial};">
                </circle>
                <text x="60" y="70" text-anchor="middle" fill="white" style="font-size: 28px; font-weight: bold;">
                    {initial_score}
                </text>
            </svg>
        </div>
        <div class="score-label">Initial Score</div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="score-card score-card-improved">', unsafe_allow_html=True)
        if updated_score != -1:
            progress_updated = updated_score / 100 * 377
            st.markdown(f'''
            <div class="progress-ring">
                <svg width="120" height="120">
                    <circle class="progress-ring-circle" cx="60" cy="60" r="50"></circle>
                    <circle class="progress-ring-circle-progress" cx="60" cy="60" r="50" 
                        style="stroke-dasharray: 377; stroke-dashoffset: {377 - progress_updated};">
                    </circle>
                    <text x="60" y="70" text-anchor="middle" fill="white" style="font-size: 28px; font-weight: bold;">
                        {updated_score}
                    </text>
                </svg>
            </div>
            <div class="score-label">Improved Score</div>
            <div class="score-delta">
                <span class="arrow-up"></span>
                <span>+{score_diff} points</span>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="score-value">N/A</div><div class="score-label">Improved Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="score-card score-card-improvement">', unsafe_allow_html=True)
        if updated_score != -1 and initial_score > 0:
            improvement_percentage = (score_diff / initial_score * 100)
            st.markdown(f'''
            <div style="margin-top: 20px;">
                <div class="score-value">{improvement_percentage:.1f}%</div>
                <div class="score-label">Total Improvement</div>
                <div class="score-delta">
                    <span class="arrow-up"></span>
                    <span>{score_diff} points gained</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="score-value">N/A</div><div class="score-label">Improvement</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Issues identified with better styling
    with st.expander("🔍 **Issues Identified**", expanded=True):
        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ffc107; color: #856404;">
            <p style="margin: 0; line-height: 1.6; font-size: 1rem;">
                {result['initial_analysis']['score_issues']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Suggestions with better styling
    with st.expander("💡 **Improvement Suggestions**", expanded=True):
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745; color: #155724;">
            <p style="margin: 0; line-height: 1.6; font-size: 1rem;">
                {result['suggested_resume_updates_description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Original sections
    st.divider()
    st.subheader("📑 Original Resume Sections")
    
    for section in result['initial_analysis']['resume_sections']:
        with st.expander(f"**{section['section_name']}**"):
            st.text(section['content'])
    
    # Suggested updated sections
    st.divider()
    st.subheader("✨ Suggested Updated Resume Sections")
    
    for section in result['suggested_updated_resume_content_json']:
        with st.expander(f"**{section['section_name']}** (Optimized)"):
            st.text(section['content'])
    
    # Download button for updated resume
    st.divider()
    updated_resume_text = ""
    for section in result['suggested_updated_resume_content_json']:
        updated_resume_text += f"{section['section_name']}:\n{section['content']}\n\n"
    
    st.download_button(
        label="📥 Download Optimized Resume (Text)",
        data=updated_resume_text,
        file_name="optimized_resume.txt",
        mime="text/plain",
        use_container_width=True
    )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>Tip:</strong> Use the suggested improvements as a guide, but ensure all information remains accurate and truthful.</p>
    <p>Made with ❤️ using Streamlit and LangChain</p>
</div>
""", unsafe_allow_html=True)
