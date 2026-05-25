import streamlit as st

st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("Resume Screening & Skill Matching")
jd = st.text_area("Job Description", height=180)
uploaded = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"],
                            accept_multiple_files=True)
st.button("Rank candidates")