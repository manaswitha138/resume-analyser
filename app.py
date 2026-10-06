import streamlit as st
import fitz
import re
import spacy
import os
import io
import pytesseract

from PIL import Image

from semantic_matcher import (
    calculate_semantic_job_match,
    match_requirements
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="CareerLens AI",
    page_icon="🎯",
    layout="wide"
)


# ==========================================================
# TESSERACT OCR CONFIGURATION
# ==========================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):

    pytesseract.pytesseract.tesseract_cmd = (
        TESSERACT_PATH
    )

    OCR_AVAILABLE = True

else:

    OCR_AVAILABLE = False


# ==========================================================
# LOAD NLP MODEL
# ==========================================================

@st.cache_resource
def load_nlp_model():

    return spacy.load(
        "en_core_web_sm"
    )


nlp = load_nlp_model()


# ==========================================================
# SKILL DATABASE
# ==========================================================

SKILLS = [
    "Python",
    "Java",
    "C",
    "C++",
    "JavaScript",
    "HTML",
    "CSS",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "Excel",
    "Power BI",
    "Tableau",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "Data Analysis",
    "Data Science",
    "Statistics",
    "Git",
    "GitHub",
    "Streamlit",
    "FastAPI",
    "Flask",
    "Django",
    "React",
    "Node.js",
    "AWS",
    "Azure",
    "Docker"
]


# ==========================================================
# PDF TEXT EXTRACTION + OCR
# ==========================================================

def extract_text_from_pdf(uploaded_file):

    pdf = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    page_count = len(pdf)

    text_parts = []

    # ------------------------------------------------------
    # NORMAL TEXT EXTRACTION
    # ------------------------------------------------------

    for page in pdf:

        page_text = page.get_text()

        if page_text:

            text_parts.append(
                page_text
            )

    text = "\n".join(
        text_parts
    ).strip()

    # ------------------------------------------------------
    # OCR FALLBACK
    # ------------------------------------------------------

    if not text:

        if not OCR_AVAILABLE:

            pdf.close()

            return "", page_count

        st.info(
            "🔍 No selectable text found. "
            "Using OCR to read the resume..."
        )

        ocr_parts = []

        progress = st.progress(
            0
        )

        for index, page in enumerate(pdf):

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(
                    2,
                    2
                ),
                alpha=False
            )

            image_bytes = pixmap.tobytes(
                "png"
            )

            image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            )

            page_text = (
                pytesseract.image_to_string(
                    image,
                    lang="eng"
                )
            )

            ocr_parts.append(
                page_text
            )

            progress.progress(
                (index + 1) / page_count
            )

        progress.empty()

        text = "\n".join(
            ocr_parts
        ).strip()

    pdf.close()

    return text, page_count


# ==========================================================
# BASIC INFORMATION
# ==========================================================

def extract_email(text):

    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    matches = re.findall(
        pattern,
        text
    )

    if matches:

        return matches[0]

    return "Not detected"


def extract_phone(text):

    patterns = [
        r"\+91[\s-]?[6-9]\d{9}",
        r"\b[6-9]\d{9}\b",
        r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        if matches:

            return matches[0]

    return "Not detected"


def extract_name(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:

        return "Not detected"

    doc = nlp(
        "\n".join(
            lines[:15]
        )
    )

    for entity in doc.ents:

        if entity.label_ == "PERSON":

            return entity.text.strip()

    for line in lines[:10]:

        if (
            len(line.split()) <= 5
            and "@" not in line
            and not any(
                char.isdigit()
                for char in line
            )
        ):

            return line

    return "Not detected"


# ==========================================================
# SECTION EXTRACTION
# ==========================================================

def extract_section(
    text,
    headings
):

    lines = text.splitlines()

    collected = []

    capturing = False

    common_sections = [
        "education",
        "experience",
        "work experience",
        "professional experience",
        "projects",
        "project",
        "skills",
        "technical skills",
        "certifications",
        "certificates",
        "achievements",
        "languages",
        "summary",
        "objective",
        "contact"
    ]

    for line in lines:

        clean_line = line.strip()

        if not clean_line:

            continue

        lower_line = (
            clean_line.lower()
        )

        if any(
            heading in lower_line
            for heading in headings
        ):

            capturing = True

            continue

        if capturing:

            if (
                lower_line in common_sections
                and not any(
                    heading in lower_line
                    for heading in headings
                )
            ):

                break

            collected.append(
                clean_line
            )

    if collected:

        return "\n".join(
            collected
        )

    return "Not detected"


# ==========================================================
# SKILL EXTRACTION
# ==========================================================

def extract_skills(text):

    detected = []

    text_lower = text.lower()

    for skill in SKILLS:

        pattern = (
            r"(?<!\w)"
            + re.escape(
                skill.lower()
            )
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            text_lower
        ):

            detected.append(
                skill
            )

    return detected


# ==========================================================
# RESUME ANALYSIS
# ==========================================================

def analyze_resume(text):

    return {

        "name": extract_name(
            text
        ),

        "email": extract_email(
            text
        ),

        "phone": extract_phone(
            text
        ),

        "skills": extract_skills(
            text
        ),

        "education": extract_section(
            text,
            [
                "education",
                "academic background",
                "academic qualification"
            ]
        ),

        "experience": extract_section(
            text,
            [
                "experience",
                "work experience",
                "professional experience",
                "internship"
            ]
        ),

        "projects": extract_section(
            text,
            [
                "projects",
                "project experience",
                "academic projects"
            ]
        ),

        "certifications": extract_section(
            text,
            [
                "certifications",
                "certificates",
                "courses"
            ]
        )
    }


# ==========================================================
# JOB REQUIREMENT ANALYSIS
# ==========================================================

def extract_job_requirements(
    job_description
):

    required_skills = []

    preferred_skills = []

    lines = job_description.splitlines()

    current_section = "general"

    required_keywords = [
        "required skills",
        "requirements",
        "required qualifications",
        "must have",
        "mandatory",
        "required"
    ]

    preferred_keywords = [
        "preferred skills",
        "preferred qualifications",
        "nice to have",
        "good to have",
        "preferred"
    ]

    for line in lines:

        clean_line = line.strip()

        if not clean_line:

            continue

        lower_line = (
            clean_line.lower()
        )

        if any(
            keyword in lower_line
            for keyword in required_keywords
        ):

            current_section = "required"

        elif any(
            keyword in lower_line
            for keyword in preferred_keywords
        ):

            current_section = "preferred"

        for skill in SKILLS:

            pattern = (
                r"(?<!\w)"
                + re.escape(
                    skill.lower()
                )
                + r"(?!\w)"
            )

            if re.search(
                pattern,
                lower_line
            ):

                if current_section == "preferred":

                    if skill not in preferred_skills:

                        preferred_skills.append(
                            skill
                        )

                else:

                    if skill not in required_skills:

                        required_skills.append(
                            skill
                        )

    return (
        required_skills,
        preferred_skills
    )


# ==========================================================
# JOB READINESS
# ==========================================================

def calculate_job_readiness(
    resume_skills,
    required_skills,
    preferred_skills
):

    resume_set = {
        skill.lower()
        for skill in resume_skills
    }

    required_matching = [
        skill
        for skill in required_skills
        if skill.lower()
        in resume_set
    ]

    required_missing = [
        skill
        for skill in required_skills
        if skill.lower()
        not in resume_set
    ]

    preferred_matching = [
        skill
        for skill in preferred_skills
        if skill.lower()
        in resume_set
    ]

    preferred_missing = [
        skill
        for skill in preferred_skills
        if skill.lower()
        not in resume_set
    ]

    required_score = (

        len(required_matching)
        / len(required_skills)

        if required_skills

        else 1
    )

    preferred_score = (

        len(preferred_matching)
        / len(preferred_skills)

        if preferred_skills

        else 1
    )

    readiness_score = round(
        (
            required_score * 75
            + preferred_score * 25
        )
    )

    return {

        "score": readiness_score,

        "required_matching":
            required_matching,

        "required_missing":
            required_missing,

        "preferred_matching":
            preferred_matching,

        "preferred_missing":
            preferred_missing
    }


# ==========================================================
# RESUME IMPROVEMENT ENGINE
# ==========================================================

def generate_resume_improvements(
    resume_skills,
    required_missing,
    preferred_missing,
    resume_data
):

    improvements = []

    if required_missing:

        improvements.append(
            "Prioritize learning these required "
            "skills: "
            + ", ".join(
                required_missing[:5]
            )
            + "."
        )

    if preferred_missing:

        improvements.append(
            "Consider adding preferred skills such as "
            + ", ".join(
                preferred_missing[:5]
            )
            + "."
        )

    if required_missing:

        improvements.append(
            "Build a project that demonstrates "
            + ", ".join(
                required_missing[:3]
            )
            + " together."
        )

    if (
        resume_data.get("projects")
        == "Not detected"
    ):

        improvements.append(
            "Add a dedicated Projects section "
            "with 2–3 relevant projects."
        )

    if (
        resume_data.get("experience")
        == "Not detected"
    ):

        improvements.append(
            "If applicable, add internships, "
            "training, or practical experience."
        )

    if (
        resume_data.get("certifications")
        == "Not detected"
    ):

        improvements.append(
            "Add relevant certifications or "
            "completed courses."
        )

    improvements.append(
        "Use job-relevant keywords naturally "
        "in Skills, Projects, and Experience."
    )

    improvements.append(
        "Where possible, describe achievements "
        "using measurable results such as "
        "accuracy, dataset size, users, or "
        "processing time."
    )

    return improvements


# ==========================================================
# ATS SCORE
# ==========================================================

def calculate_ats_score(
    resume_text,
    resume_data,
    required_skills,
    preferred_skills
):

    score = 0

    checks = {}

    # ------------------------------------------------------
    # CONTACT
    # ------------------------------------------------------

    contact_score = 0

    if (
        resume_data["email"]
        != "Not detected"
    ):

        contact_score += 5

    if (
        resume_data["phone"]
        != "Not detected"
    ):

        contact_score += 5

    score += contact_score

    checks[
        "Contact Information"
    ] = contact_score

    # ------------------------------------------------------
    # SKILLS
    # ------------------------------------------------------

    skills_score = (
        20
        if resume_data["skills"]
        else 0
    )

    score += skills_score

    checks[
        "Skills Section"
    ] = skills_score

    # ------------------------------------------------------
    # EDUCATION
    # ------------------------------------------------------

    education_score = (

        0
        if resume_data["education"]
        == "Not detected"

        else 15
    )

    score += education_score

    checks[
        "Education"
    ] = education_score

    # ------------------------------------------------------
    # PROJECTS
    # ------------------------------------------------------

    projects_score = (

        0
        if resume_data["projects"]
        == "Not detected"

        else 15
    )

    score += projects_score

    checks[
        "Projects"
    ] = projects_score

    # ------------------------------------------------------
    # EXPERIENCE
    # ------------------------------------------------------

    experience_score = (

        0
        if resume_data["experience"]
        == "Not detected"

        else 15
    )

    score += experience_score

    checks[
        "Experience"
    ] = experience_score

    # ------------------------------------------------------
    # CERTIFICATIONS
    # ------------------------------------------------------

    certifications_score = (

        0
        if resume_data["certifications"]
        == "Not detected"

        else 10
    )

    score += certifications_score

    checks[
        "Certifications"
    ] = certifications_score

    # ------------------------------------------------------
    # JOB KEYWORDS
    # ------------------------------------------------------

    job_skills = list(
        dict.fromkeys(
            required_skills
            + preferred_skills
        )
    )

    resume_skill_names = {
        skill.lower()
        for skill in resume_data["skills"]
    }

    matching_keywords = [
        skill
        for skill in job_skills
        if skill.lower()
        in resume_skill_names
    ]

    if job_skills:

        keyword_percentage = (
            len(matching_keywords)
            / len(job_skills)
        )

        keyword_score = round(
            keyword_percentage * 15
        )

    else:

        keyword_score = 0

    score += keyword_score

    checks[
        "Job Keywords"
    ] = keyword_score

    return {

        "score": min(
            score,
            100
        ),

        "checks": checks,

        "matching_keywords":
            matching_keywords,

        "total_job_keywords":
            len(job_skills)
    }


# ==========================================================
# DISPLAY RESUME PROFILE
# ==========================================================

def display_resume_profile(
    resume_data
):

    st.subheader(
        "👤 Resume Profile"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Name**"
        )

        st.info(
            resume_data["name"]
        )

        st.write(
            "**Email**"
        )

        st.info(
            resume_data["email"]
        )

    with col2:

        st.write(
            "**Phone**"
        )

        st.info(
            resume_data["phone"]
        )

        st.write(
            "**Detected Skills**"
        )

        st.info(
            str(
                len(
                    resume_data["skills"]
                )
            )
        )


# ==========================================================
# DISPLAY SKILLS
# ==========================================================

def display_skills(
    resume_data
):

    st.subheader(
        "🛠️ Skills Detected"
    )

    if resume_data["skills"]:

        columns = st.columns(4)

        for index, skill in enumerate(
            resume_data["skills"]
        ):

            with columns[
                index % 4
            ]:

                st.success(
                    f"✓ {skill}"
                )

    else:

        st.warning(
            "No predefined skills detected."
        )


# ==========================================================
# DISPLAY JOB ANALYSIS
# ==========================================================

def display_job_analysis(
    analysis,
    required_skills,
    preferred_skills
):

    score = analysis["score"]

    st.subheader(
        "🎯 Job Readiness"
    )

    score_col, status_col = (
        st.columns([1, 2])
    )

    with score_col:

        st.metric(
            "Readiness Score",
            f"{score}%"
        )

        st.progress(
            score / 100
        )

    with status_col:

        if score >= 80:

            st.success(
                "🟢 Strong skill alignment"
            )

        elif score >= 60:

            st.warning(
                "🟡 Moderate skill alignment"
            )

        else:

            st.error(
                "🔴 Significant skill gaps"
            )

        st.write(
            "This score measures alignment between "
            "detected resume skills and detected "
            "job requirements."
        )

    # ------------------------------------------------------
    # REQUIRED
    # ------------------------------------------------------

    st.subheader(
        "🔴 Required Skills"
    )

    if required_skills:

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "### ✅ You Have"
            )

            if analysis[
                "required_matching"
            ]:

                for skill in analysis[
                    "required_matching"
                ]:

                    st.success(
                        f"✓ {skill}"
                    )

            else:

                st.info(
                    "No required skills matched."
                )

        with col2:

            st.write(
                "### ⚠️ Missing"
            )

            if analysis[
                "required_missing"
            ]:

                for skill in analysis[
                    "required_missing"
                ]:

                    st.error(
                        f"⚠ {skill}"
                    )

            else:

                st.success(
                    "All detected required skills "
                    "are present!"
                )

    else:

        st.info(
            "No clearly required skills detected."
        )

    # ------------------------------------------------------
    # PREFERRED
    # ------------------------------------------------------

    st.subheader(
        "🟡 Preferred Skills"
    )

    if preferred_skills:

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "### ✅ You Have"
            )

            if analysis[
                "preferred_matching"
            ]:

                for skill in analysis[
                    "preferred_matching"
                ]:

                    st.success(
                        f"✓ {skill}"
                    )

            else:

                st.info(
                    "No preferred skills matched."
                )

        with col2:

            st.write(
                "### 💡 Could Improve"
            )

            if analysis[
                "preferred_missing"
            ]:

                for skill in analysis[
                    "preferred_missing"
                ]:

                    st.warning(
                        f"○ {skill}"
                    )

            else:

                st.success(
                    "All detected preferred skills "
                    "are present!"
                )

    else:

        st.info(
            "No preferred skills detected."
        )

    # ------------------------------------------------------
    # CRITICAL GAPS
    # ------------------------------------------------------

    st.subheader(
        "🚨 Critical Skill Gaps"
    )

    critical_gaps = (
        analysis["required_missing"]
    )

    if critical_gaps:

        st.error(
            "The following required skills were "
            "not detected:"
        )

        for skill in critical_gaps:

            st.write(
                f"🔴 **{skill}**"
            )

    else:

        st.success(
            "🎉 No critical required-skill gaps detected."
        )


# ==========================================================
# DISPLAY RESUME IMPROVEMENTS
# ==========================================================

def display_resume_improvements(
    resume_data,
    analysis,
    required_skills,
    preferred_skills
):

    st.header(
        "🤖 Step 4 — Resume Improvement"
    )

    improvements = (
        generate_resume_improvements(
            resume_data["skills"],
            analysis["required_missing"],
            analysis["preferred_missing"],
            resume_data
        )
    )

    st.subheader(
        "📈 Skills to Improve"
    )

    skills_to_improve = (
        analysis["required_missing"]
        + analysis["preferred_missing"]
    )

    if skills_to_improve:

        for index, skill in enumerate(
            skills_to_improve[:5],
            start=1
        ):

            st.write(
                f"**{index}. {skill}**"
            )

    else:

        st.success(
            "🎉 No major skill gaps detected."
        )

    st.subheader(
        "📝 Resume Improvement Suggestions"
    )

    for suggestion in improvements:

        st.info(
            "💡 " + suggestion
        )

    st.subheader(
        "🔑 ATS Keywords"
    )

    ats_keywords = list(
        dict.fromkeys(
            required_skills
            + preferred_skills
        )
    )

    if ats_keywords:

        st.write(
            "Use relevant keywords naturally "
            "in your resume:"
        )

        st.code(
            " | ".join(
                ats_keywords
            )
        )

    st.subheader(
        "🚀 Recommended Next Step"
    )

    if analysis[
        "required_missing"
    ]:

        top_gaps = (
            analysis[
                "required_missing"
            ][:3]
        )

        st.success(
            "Build a project that demonstrates: "
            + ", ".join(
                top_gaps
            )
            + "."
        )

    else:

        st.success(
            "Your detected skills align well "
            "with the target job."
        )


# ==========================================================
# DISPLAY ATS ANALYSIS
# ==========================================================

def display_ats_analysis(
    resume_text,
    resume_data,
    required_skills,
    preferred_skills
):

    st.header(
        "📊 Step 5 — ATS Resume Score"
    )

    ats_analysis = (
        calculate_ats_score(
            resume_text,
            resume_data,
            required_skills,
            preferred_skills
        )
    )

    ats_score = (
        ats_analysis["score"]
    )

    st.subheader(
        "🎯 Overall ATS Score"
    )

    score_col, status_col = (
        st.columns([1, 2])
    )

    with score_col:

        st.metric(
            "ATS Score",
            f"{ats_score}/100"
        )

        st.progress(
            ats_score / 100
        )

    with status_col:

        if ats_score >= 80:

            st.success(
                "🟢 Strong ATS readiness"
            )

        elif ats_score >= 60:

            st.warning(
                "🟡 Moderate ATS readiness"
            )

        else:

            st.error(
                "🔴 Resume needs improvement"
            )

    # ------------------------------------------------------
    # CATEGORY BREAKDOWN
    # ------------------------------------------------------

    st.subheader(
        "📋 ATS Category Breakdown"
    )

    categories = (
        ats_analysis["checks"]
    )

    col1, col2 = st.columns(2)

    items = list(
        categories.items()
    )

    for index, (
        category,
        points
    ) in enumerate(items):

        target = (
            col1
            if index % 2 == 0
            else col2
        )

        with target:

            if points >= 15:

                st.success(
                    f"✅ {category}: "
                    f"{points} points"
                )

            elif points >= 8:

                st.warning(
                    f"⚠️ {category}: "
                    f"{points} points"
                )

            else:

                st.error(
                    f"❌ {category}: "
                    f"{points} points"
                )

    # ------------------------------------------------------
    # JOB KEYWORDS
    # ------------------------------------------------------

    st.subheader(
        "🔑 Job Keyword Analysis"
    )

    total_keywords = (
        ats_analysis[
            "total_job_keywords"
        ]
    )

    matching_keywords = (
        ats_analysis[
            "matching_keywords"
        ]
    )

    keyword_percentage = 0

    if total_keywords:

        keyword_percentage = round(
            (
                len(matching_keywords)
                / total_keywords
            ) * 100
        )

        st.write(
            f"Your resume matches "
            f"**{keyword_percentage}%** "
            f"of detected job-related skills."
        )

        if matching_keywords:

            st.write(
                "### ✅ Matching Keywords"
            )

            for skill in matching_keywords:

                st.success(
                    f"✓ {skill}"
                )

    else:

        st.info(
            "No job-specific keywords detected."
        )

    # ------------------------------------------------------
    # ATS SUGGESTIONS
    # ------------------------------------------------------

    st.subheader(
        "💡 ATS Improvement Suggestions"
    )

    suggestions = []

    if (
        ats_analysis["checks"][
            "Contact Information"
        ] < 10
    ):

        suggestions.append(
            "Make sure your email and phone "
            "number are clearly visible."
        )

    if (
        ats_analysis["checks"][
            "Skills Section"
        ] == 0
    ):

        suggestions.append(
            "Add a clearly labeled "
            "Technical Skills section."
        )

    if (
        ats_analysis["checks"][
            "Education"
        ] == 0
    ):

        suggestions.append(
            "Add an Education section with "
            "your degree and academic details."
        )

    if (
        ats_analysis["checks"][
            "Projects"
        ] == 0
    ):

        suggestions.append(
            "Add relevant projects that "
            "demonstrate target-job skills."
        )

    if (
        ats_analysis["checks"][
            "Experience"
        ] == 0
    ):

        suggestions.append(
            "Add internships, work experience, "
            "or practical experience if applicable."
        )

    if (
        ats_analysis["checks"][
            "Certifications"
        ] == 0
    ):

        suggestions.append(
            "Add relevant certifications or "
            "completed courses."
        )

    if (
        total_keywords
        and keyword_percentage < 70
    ):

        suggestions.append(
            "Increase relevant job-description "
            "keywords naturally throughout "
            "your resume."
        )

    if not suggestions:

        suggestions.append(
            "Your resume has a strong ATS "
            "structure. Continue tailoring "
            "keywords and achievements."
        )

    for suggestion in suggestions:

        st.info(
            "💡 " + suggestion
        )

    # ------------------------------------------------------
    # SCORE CALCULATION
    # ------------------------------------------------------

    with st.expander(
        "📊 How the ATS Score Is Calculated"
    ):

        st.write(
            "Contact Information: **10 points**"
        )

        st.write(
            "Skills Section: **20 points**"
        )

        st.write(
            "Education: **15 points**"
        )

        st.write(
            "Projects: **15 points**"
        )

        st.write(
            "Experience: **15 points**"
        )

        st.write(
            "Certifications: **10 points**"
        )

        st.write(
            "Job Keyword Alignment: **15 points**"
        )

        st.write(
            "**Total: 100 points**"
        )

    return ats_score


# ==========================================================
# DISPLAY SEMANTIC ANALYSIS
# ==========================================================

def display_semantic_analysis(
    resume_text,
    job_description,
    required_skills,
    preferred_skills
):

    st.header(
        "🧠 Step 6 — Semantic Matching"
    )

    with st.spinner(
        "Analyzing semantic similarity..."
    ):

        semantic_result = (
            calculate_semantic_job_match(
                resume_text,
                job_description
            )
        )

    semantic_score = (
        semantic_result["score"]
    )

    semantic_level = (
        semantic_result["level"]
    )

    score_col, status_col = (
        st.columns([1, 2])
    )

    with score_col:

        st.metric(
            "Semantic Match",
            f"{semantic_score}%"
        )

        st.progress(
            semantic_score / 100
        )

    with status_col:

        if semantic_score >= 80:

            st.success(
                f"🟢 {semantic_level}"
            )

        elif semantic_score >= 60:

            st.warning(
                f"🟡 {semantic_level}"
            )

        else:

            st.error(
                f"🔴 {semantic_level}"
            )

    # ------------------------------------------------------
    # REQUIREMENT EVIDENCE
    # ------------------------------------------------------

    st.subheader(
        "🔍 Requirement Evidence"
    )

    job_requirements = list(
        dict.fromkeys(
            required_skills
            + preferred_skills
        )
    )

    if job_requirements:

        with st.spinner(
            "Finding resume evidence..."
        ):

            semantic_matches = (
                match_requirements(
                    resume_text,
                    job_requirements
                )
            )

        for result in semantic_matches:

            requirement = (
                result["requirement"]
            )

            similarity = (
                result["similarity"]
            )

            evidence = (
                result["evidence"]
            )

            status = (
                result["status"]
            )

            if status == "Strong Match":

                st.success(
                    f"✅ {requirement} — "
                    f"{similarity}% match"
                )

                if evidence:

                    st.caption(
                        "Resume evidence: "
                        + evidence
                    )

            else:

                st.warning(
                    f"⚠️ {requirement} — "
                    f"{similarity}% similarity"
                )

                if evidence:

                    st.caption(
                        "Closest resume evidence: "
                        + evidence
                    )

    else:

        st.info(
            "No job requirements detected."
        )

    return semantic_score


# ==========================================================
# APPLICATION
# ==========================================================

st.title(
    "🎯 CareerLens AI"
)

st.write(
    "AI-powered resume analysis, ATS scoring, "
    "job matching, and semantic skill analysis."
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "📌 CareerLens AI"
)

st.sidebar.write(
    "Your resume intelligence and job "
    "matching assistant."
)

st.sidebar.info(
    "Upload a PDF resume and paste a "
    "job description to begin."
)


# ==========================================================
# STEP 1 — RESUME UPLOAD
# ==========================================================

st.header(
    "📄 Step 1 — Upload Resume"
)

uploaded_file = st.file_uploader(
    "Choose your resume PDF",
    type=["pdf"]
)


if not uploaded_file:

    st.info(
        "👆 Upload a PDF resume to begin."
    )

    st.stop()


# ==========================================================
# EXTRACT RESUME
# ==========================================================

resume_text, page_count = (
    extract_text_from_pdf(
        uploaded_file
    )
)


if not resume_text.strip():

    st.error(
        "❌ No readable text was found "
        "in this PDF."
    )

    if not OCR_AVAILABLE:

        st.error(
            "Tesseract OCR is not available. "
            "Check your Tesseract installation."
        )

    st.stop()


# ==========================================================
# ANALYZE RESUME
# ==========================================================

resume_data = analyze_resume(
    resume_text
)

st.success(
    "✅ Resume successfully analyzed!"
)


# ==========================================================
# RESUME OVERVIEW
# ==========================================================

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Pages",
        page_count
    )

with col2:

    st.metric(
        "Words",
        len(
            resume_text.split()
        )
    )

with col3:

    st.metric(
        "Skills",
        len(
            resume_data["skills"]
        )
    )

with col4:

    st.metric(
        "Characters",
        len(resume_text)
    )


# ==========================================================
# PROFILE
# ==========================================================

display_resume_profile(
    resume_data
)


# ==========================================================
# SKILLS
# ==========================================================

display_skills(
    resume_data
)


# ==========================================================
# STEP 2 — TARGET JOB
# ==========================================================

st.header(
    "💼 Step 2 — Target Job"
)

job_description = st.text_area(
    "Paste the complete job description here",
    height=300,
    placeholder=(
        "Example:\n\n"
        "We are looking for a Data Analyst "
        "with experience in Python, SQL, "
        "Excel and Power BI..."
    )
)


if not job_description.strip():

    st.info(
        "👆 Paste a job description to continue."
    )

    st.stop()


# ==========================================================
# STEP 3 — JOB ANALYSIS
# ==========================================================

st.header(
    "🔎 Step 3 — Job Analysis"
)

required_skills, preferred_skills = (
    extract_job_requirements(
        job_description
    )
)


if (
    not required_skills
    and not preferred_skills
):

    st.warning(
        "No skills from the current skill "
        "database were detected."
    )

    st.info(
        "Try a job description containing "
        "skills such as Python, SQL, Excel, "
        "Power BI, Pandas, Machine Learning, "
        "Statistics, etc."
    )

    st.stop()


# ==========================================================
# JOB READINESS
# ==========================================================

analysis = calculate_job_readiness(
    resume_data["skills"],
    required_skills,
    preferred_skills
)


display_job_analysis(
    analysis,
    required_skills,
    preferred_skills
)


# ==========================================================
# STEP 4 — RESUME IMPROVEMENT
# ==========================================================

display_resume_improvements(
    resume_data,
    analysis,
    required_skills,
    preferred_skills
)


# ==========================================================
# STEP 5 — ATS SCORE
# ==========================================================

ats_score = display_ats_analysis(
    resume_text,
    resume_data,
    required_skills,
    preferred_skills
)


# ==========================================================
# STEP 6 — SEMANTIC MATCHING
# ==========================================================

semantic_score = display_semantic_analysis(
    resume_text,
    job_description,
    required_skills,
    preferred_skills
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

st.header(
    "🏆 CareerLens AI — Final Analysis"
)

final_col1, final_col2, final_col3 = (
    st.columns(3)
)

with final_col1:

    st.metric(
        "🎯 Job Readiness",
        f"{analysis['score']}%"
    )

with final_col2:

    st.metric(
        "📊 ATS Score",
        f"{ats_score}/100"
    )

with final_col3:

    st.metric(
        "🧠 Semantic Match",
        f"{semantic_score}%"
    )


# ==========================================================
# FINAL INTERPRETATION
# ==========================================================

overall_score = round(
    (
        analysis["score"] * 0.40
        + ats_score * 0.30
        + semantic_score * 0.30
    )
)


st.subheader(
    "⭐ Overall CareerLens Score"
)

st.metric(
    "Overall Score",
    f"{overall_score}%"
)

st.progress(
    overall_score / 100
)


if overall_score >= 80:

    st.success(
        "🟢 Strong overall alignment. "
        "Your resume has good job-skill alignment, "
        "ATS structure, and semantic relevance."
    )

elif overall_score >= 60:

    st.warning(
        "🟡 Moderate overall alignment. "
        "Your resume has a reasonable foundation, "
        "but there are areas worth improving."
    )

else:

    st.error(
        "🔴 Significant improvement opportunity. "
        "Focus on the identified skill gaps, "
        "resume structure, and job relevance."
    )


st.caption(
    "CareerLens AI scores are analytical indicators "
    "based on extracted resume information, detected "
    "skills, ATS-style checks, and semantic similarity. "
    "They are not predictions of hiring probability "
    "and do not guarantee an interview or job offer."
)