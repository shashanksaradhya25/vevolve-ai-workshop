from crewai import Agent

gap_agent = Agent(
    role="Skill Gap Analyzer",
    goal="Find missing skills",
    backstory="Career coach",
    verbose=True
)