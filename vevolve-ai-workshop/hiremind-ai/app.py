# =====================================================
# app.py (FULL UPDATED VERSION)
# =====================================================

import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from openai import OpenAI

from utils.loader import load_pdf
from utils.skills import extract_skills
from utils.resources import get_resources

from rag.qdrant_store import init_db, add_text, search

from crewai import Task, Crew
from agents.resume_agent import resume_agent
from agents.jd_agent import jd_agent
from agents.gap_agent import gap_agent
from agents.interview_agent import interview_agent

# =====================================================
# ENV
# =====================================================

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="HireMind AI",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 100%;
}

h1, h2, h3 {
    color: white;
}

[data-testid="stExpander"] {
    border-radius: 15px;
    border: 1px solid #2d2d2d;
    margin-bottom: 10px;
    background-color: #161B22;
}

.stMetric {
    background-color: #161B22;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("🎯 HireMind AI")

st.markdown("""
### 🚀 AI-Powered Interview Preparation Platform

Analyze resumes, identify skill gaps, practice mock interviews,
and get personalized preparation guidance using AI Agents.
""")

# =====================================================
# VECTOR DB
# =====================================================

init_db()

# =====================================================
# SESSION STATE
# =====================================================

if "resume_skills" not in st.session_state:
    st.session_state.resume_skills = []

if "jd_skills" not in st.session_state:
    st.session_state.jd_skills = []

if "missing_skills" not in st.session_state:
    st.session_state.missing_skills = []

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "questions" not in st.session_state:
    st.session_state.questions = []

if "q_index" not in st.session_state:
    st.session_state.q_index = 0

if "answers" not in st.session_state:
    st.session_state.answers = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# =====================================================
# FUNCTIONS
# =====================================================

def calculate_score(resume_skills, jd_skills):

    matched = len(set(resume_skills) & set(jd_skills))
    total = len(jd_skills)

    if total == 0:
        return 0

    return min(int((matched / total) * 100), 100)

# =====================================================

def evaluate_answer(question, answer):

    prompt = f"""
You are an expert technical interviewer.

Question:
{question}

Candidate Answer:
{answer}

Evaluate the answer.

Return ONLY valid JSON.

Format:
{{
  "correctness": "Correct / Partially Correct / Incorrect",
  "score": 0-10,
  "feedback": "short feedback",
  "ideal_answer": "best possible answer"
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    try:

        return json.loads(
            res.choices[0].message.content
        )

    except:

        return {
            "correctness": "Partially Correct",
            "score": 5,
            "feedback": "Could not fully evaluate answer.",
            "ideal_answer": "Explain with better technical depth."
        }

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "📄 Resume Analyzer",
    "🎤 Mock Interview",
    "📚 Learning Path"
])

# =====================================================
# TAB 1 - RESUME ANALYZER
# =====================================================

with tab1:

    st.header("📄 Resume Analyzer")

    col1, col2 = st.columns(2)

    with col1:

        resume = st.file_uploader(
            "Upload Resume (PDF)"
        )

    with col2:

        jd = st.text_area(
            "Paste Job Description",
            height=250
        )

    # =====================================================

    if st.button("🚀 Analyze Resume"):

        if resume is None or jd.strip() == "":

            st.warning(
                "Please upload resume and paste JD"
            )

        else:

            # =====================================================
            # LOAD RESUME
            # =====================================================

            resume_text = load_pdf(resume)

            # =====================================================
            # EXTRACT SKILLS
            # =====================================================

            resume_skills = extract_skills(
                resume_text
            )

            jd_skills = extract_skills(jd)

            missing = list(
                set(jd_skills) - set(resume_skills)
            )

            # =====================================================
            # SAVE SESSION
            # =====================================================

            st.session_state.resume_text = resume_text
            st.session_state.resume_skills = resume_skills
            st.session_state.jd_skills = jd_skills
            st.session_state.missing_skills = missing

            # =====================================================
            # VECTOR DB
            # =====================================================

            add_text(resume_text)
            add_text(jd)

            context = search(jd)

            # =====================================================
            # CREW AI TASKS
            # =====================================================

            tasks = [

                Task(
                    description=f"Resume skills: {resume_skills}",
                    expected_output="Skill summary",
                    agent=resume_agent
                ),

                Task(
                    description=f"JD skills: {jd_skills}",
                    expected_output="Required skills",
                    agent=jd_agent
                ),

                Task(
                    description=f"Missing skills: {missing}",
                    expected_output="Gap analysis",
                    agent=gap_agent
                ),

                Task(
                    description=f"Generate interview preparation guidance based on {context}",
                    expected_output="Preparation roadmap",
                    agent=interview_agent
                )
            ]

            crew = Crew(
                agents=[
                    resume_agent,
                    jd_agent,
                    gap_agent,
                    interview_agent
                ],
                tasks=tasks,
                verbose=False
            )

            with st.spinner(
                "🤖 AI analyzing resume..."
            ):

                crew.kickoff()

            # =====================================================
            # RESULTS
            # =====================================================

            st.markdown("---")

            st.subheader("📊 Results Dashboard")

            score = calculate_score(
                resume_skills,
                jd_skills
            )

            st.progress(score / 100)

            # =====================================================
            # METRICS
            # =====================================================

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "✅ Match Score",
                    f"{score}%"
                )

            with c2:

                st.metric(
                    "📄 Resume Skills",
                    len(resume_skills)
                )

            with c3:

                st.metric(
                    "⚠ Missing Skills",
                    len(missing)
                )

            # =====================================================
            # CHARTS
            # =====================================================

            st.markdown("## 📈 Skill Analytics")

            chart1, chart2 = st.columns(2)

            # PIE CHART

            with chart1:

                matched = len(
                    set(resume_skills) &
                    set(jd_skills)
                )

                pie_df = pd.DataFrame({
                    "Category": [
                        "Matched",
                        "Missing"
                    ],
                    "Count": [
                        matched,
                        len(missing)
                    ]
                })

                fig = px.pie(
                    pie_df,
                    names="Category",
                    values="Count",
                    hole=0.5,
                    color_discrete_sequence=[
                        "#00CC96",
                        "#EF553B"
                    ]
                )

                fig.update_layout(
                    paper_bgcolor="#0E1117",
                    font_color="white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # BAR CHART

            with chart2:

                bar_df = pd.DataFrame({
                    "Type": [
                        "Resume",
                        "JD",
                        "Missing"
                    ],
                    "Count": [
                        len(resume_skills),
                        len(jd_skills),
                        len(missing)
                    ]
                })

                fig2 = px.bar(
                    bar_df,
                    x="Type",
                    y="Count",
                    color="Type"
                )

                fig2.update_layout(
                    paper_bgcolor="#0E1117",
                    font_color="white"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

            # =====================================================
            # SKILLS DISPLAY
            # =====================================================

            left, right = st.columns(2)

            with left:

                st.markdown("## 🧑 Resume Skills")

                for s in resume_skills:
                    st.success(s.upper())

            with right:

                st.markdown("## 📄 JD Skills")

                for s in jd_skills:
                    st.info(s.upper())

            # =====================================================
            # MISSING SKILLS
            # =====================================================

            st.markdown("## ⚠ Missing Skills")

            if missing:

                for s in missing:
                    st.error(s.upper())

            else:

                st.success(
                    "No Missing Skills 🎉"
                )

            # =====================================================
            # INTERVIEW PREPARATION
            # =====================================================

            st.markdown("---")

            st.markdown(
                "# 🎯 Interview Preparation Guide"
            )

            matched_skills = list(
                set(resume_skills) &
                set(jd_skills)
            )

            for skill in matched_skills:

                with st.expander(
                    f"🚀 {skill.upper()}"
                ):

                    st.markdown(f"""
### Important Topics
- Fundamentals of {skill}
- Real-world scenarios
- Troubleshooting
- Architecture understanding

### Preparation Tips
✅ Explain projects using {skill}

✅ Mention scalability improvements

✅ Discuss deployment challenges

✅ Highlight production experience
""")

            # =====================================================
            # LEARNING PATH
            # =====================================================

            if missing:

                st.markdown(
                    "## 📚 Skills To Improve"
                )

                for skill in missing:

                    with st.expander(
                        f"📘 Learn {skill.upper()}"
                    ):

                        res = get_resources(skill)

                        st.write("🎥 YouTube:")
                        st.write(res["youtube"])

                        st.write("📄 Docs:")
                        st.write(res["docs"])

            st.warning(
                "👉 Go to Mock Interview tab next."
            )

# =====================================================
# TAB 2 - MOCK INTERVIEW
# =====================================================

with tab2:

    st.header("🎤 AI Mock Interview")

    resume_skills = st.session_state.resume_skills
    jd_skills = st.session_state.jd_skills
    missing = st.session_state.missing_skills
    resume_text = st.session_state.resume_text

    if not resume_skills:

        st.warning(
            "⚠ Please analyze resume first."
        )

    else:

        st.markdown("""
### 🤖 Personalized AI Interview

Questions are dynamically generated using:
- Resume skills
- Job description
- Missing skills
- Candidate profile
""")

        # =====================================================
        # GENERATE QUESTIONS
        # =====================================================

        if st.button("🚀 Generate AI Interview"):

            with st.spinner(
                "🤖 Generating questions..."
            ):

                prompt = f"""
You are an expert technical interviewer.

Resume Skills:
{resume_skills}

JD Skills:
{jd_skills}

Missing Skills:
{missing}

Resume:
{resume_text}

Generate 5 interview questions.

Requirements:
- technical
- behavioral
- scenario-based

Return ONLY python list format.
"""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                try:

                    questions = eval(
                        response.choices[0]
                        .message.content
                    )

                except:

                    questions = [
                        "Explain Kubernetes experience",
                        "How did you use AWS?",
                        "Describe a challenging project",
                        "How do you debug production issues?",
                        "Explain AI project experience"
                    ]

                st.session_state.questions = questions
                st.session_state.q_index = 0
                st.session_state.answers = []
                st.session_state.feedback = []
                st.session_state.input_key = 0

                st.success(
                    "✅ Interview Generated"
                )

        # =====================================================
        # QUESTIONS FLOW
        # =====================================================

        if st.session_state.questions:

            idx = st.session_state.q_index

            if idx < len(
                st.session_state.questions
            ):

                question = st.session_state.questions[idx]

                progress = (
                    (idx + 1)
                    /
                    len(st.session_state.questions)
                )

                st.progress(progress)

                st.markdown(f"""
# 📌 Question {idx+1}

### {question}
""")

                answer = st.text_area(
                    "✍ Your Answer",
                    height=200,
                    key=f"answer_{st.session_state.input_key}"
                )

                if st.button("✅ Submit Answer"):

                    if answer.strip() == "":

                        st.warning(
                            "Please enter answer."
                        )

                    else:

                        with st.spinner(
                            "🧠 Evaluating answer..."
                        ):

                            evaluation = evaluate_answer(
                                question,
                                answer
                            )

                        st.session_state.answers.append(
                            answer
                        )

                        st.session_state.feedback.append({
                            "question": question,
                            "answer": answer,
                            "evaluation": evaluation
                        })

                        st.session_state.q_index += 1
                        st.session_state.input_key += 1

                        st.rerun()

            # =====================================================
            # FINAL REPORT
            # =====================================================

            else:

                st.balloons()

                st.success(
                    "🎉 Interview Completed!"
                )

                total = sum(
                    item["evaluation"]["score"]
                    for item in st.session_state.feedback
                )

                avg = total / len(
                    st.session_state.feedback
                )

                st.markdown(
                    "# 🏆 Final Interview Report"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "⭐ Score",
                        f"{avg:.1f}/10"
                    )

                with c2:

                    if avg >= 8:
                        st.success("Excellent")

                    elif avg >= 6:
                        st.warning("Good")

                    else:
                        st.error("Needs Improvement")

                with c3:

                    st.metric(
                        "📌 Questions",
                        len(st.session_state.questions)
                    )

                st.progress(avg / 10)

                st.markdown("---")

                # =====================================================
                # FEEDBACK PER QUESTION
                # =====================================================

                for i, item in enumerate(
                    st.session_state.feedback
                ):

                    eval_data = item["evaluation"]

                    with st.expander(
                        f"📌 Question {i+1}"
                    ):

                        st.markdown(
                            f"### ❓ {item['question']}"
                        )

                        st.markdown(
                            f"### 👤 Your Answer\n\n{item['answer']}"
                        )

                        st.metric(
                            "⭐ Score",
                            f"{eval_data['score']}/10"
                        )

                        st.metric(
                            "✔ Correctness",
                            eval_data["correctness"]
                        )

                        st.info(
                            eval_data["feedback"]
                        )

                        st.success(
                            eval_data["ideal_answer"]
                        )

                st.markdown("---")

                st.markdown(
                    "## 🎯 Final Suggestions"
                )

                if missing:

                    for skill in missing:

                        st.warning(
                            f"Improve {skill.upper()}"
                        )

                st.info("""
✅ Use STAR Method

✅ Explain projects clearly

✅ Focus on scalability

✅ Mention production challenges
""")

# =====================================================
# TAB 3 - LEARNING PATH
# =====================================================

with tab3:

    st.header("📚 Personalized Learning Path")

    missing = st.session_state.missing_skills

    if not missing:

        st.success(
            "No missing skills found 🎉"
        )

    else:

        for skill in missing:

            with st.expander(
                f"🚀 Learn {skill.upper()}"
            ):

                res = get_resources(skill)

                st.write("🎥 YouTube")
                st.write(res["youtube"])

                st.write("📘 Documentation")
                st.write(res["docs"])

                st.markdown("""
### Suggested Plan

Week 1:
- Learn fundamentals

Week 2:
- Build mini project

Week 3:
- Practice interview questions

Week 4:
- Deploy & revise
""")