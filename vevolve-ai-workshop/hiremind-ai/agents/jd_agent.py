from crewai import Agent

jd_agent = Agent(
    role="JD Analyzer",
    goal="Extract skills from job description",
    backstory="Recruiter",
    verbose=True
)