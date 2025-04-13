from crewai_tools import CSVSearchTool, SerperDevTool
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os
from crewai.llm import LLM


# Load environment variables
load_dotenv()

# Initialize tools
csv_tool = CSVSearchTool(csv='C:/Users/ayush/new_crewai_project/lapse_researcher/src/lapse_researcher/artifacts/chunks/chunk_0.csv')
search_tool = SerperDevTool()

# Create agents
csv_agent = Agent(
    role="Insurance Data Analyst",
    goal="Analyze CSV data to identify lapse patterns and risk factors",
    backstory="Expert in insurance data analysis with deep understanding of actuarial metrics",
    tools=[csv_tool],
    verbose=True,
    allow_delegation=True,
)

market_analyst = Agent(
    role="Market Intelligence Analyst",
    goal="Research industry trends and competitive landscape",
    backstory="Skilled in financial market research and strategic analysis",
    tools=[search_tool],
    verbose=True,
    allow_delegation=True,
)

# Create task with enhanced requirements
strategic_analysis_task = Task(
    description="""Analyze {question} using both CSV data and current market trends.
    1. First query the CSV file for specific metrics.
    2. Then research industry benchmarks and recent developments.
    3. Combine insights to create strategic recommendations.""",
    expected_output="""Comprehensive report containing:
    - CSV data findings.
    - Market context analysis.
    - Actionable strategies with risk assessment.
    - Implementation roadmap.""",
    agents=[csv_agent, market_analyst],
    async_execution=False,
)

# Configure CrewAI crew
strategic_crew = Crew(
    agents=[csv_agent, market_analyst],
    tasks=[strategic_analysis_task],
    manager_llm=LLM(model="gpt-4"),
    process=Process.hierarchical,
    verbose=True,
)

# Initialize Slack client
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
slack_channel = os.getenv("SLACK_CHANNEL_ID")

def send_message_to_slack(message):
    """
    Sends a message to a Slack channel using the Slack API.
    
    :param message: The message text to send to Slack.
    """
    try:
        response = slack_client.chat_postMessage(channel=slack_channel, text=message)
        print(f"Message sent to Slack: {response['message']['text']}")
    except SlackApiError as e:
        print(f"Error sending message to Slack: {e.response['error']}")

# Execution loop
while True:
    try:
        question = input("\nEnter strategic question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        # Run CrewAI process
        result = strategic_crew.kickoff(inputs={"question": question})

        # Print result locally and send it to Slack
        print(f"\nStrategic Analysis:\n{'-'*30}\n{result}")
        send_message_to_slack(f"Strategic Analysis Result:\n{result}")

    except Exception as e:
        print(f"Error processing request: {str(e)}")
