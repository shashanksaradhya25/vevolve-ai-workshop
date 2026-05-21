from crewai import Agent

interview_agent = Agent(
    role="Interview Coach",
    goal="Ask questions and evaluate answers",
    backstory="FAANG interviewer",
    verbose=True
)