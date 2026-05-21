from crewai import Agent

resume_agent = Agent(
    role="Resume Analyzer",
    goal="Extract skills from resume",
    backstory="HR expert",
    verbose=True
)