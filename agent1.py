from langchain.agents import create_agent
from dotenv import load_dotenv

userReqInput = input("Please enter your requirement: ")
testType = input("Please enter test Type (positive/negative/all): ")
noOfTestCases = input("Please enter number of test cases: ")

System_prompt = """You are a Senior QA Engineer person.Your responsibility is to analyse teh requirement and generate
 test cases based on the  user input.
Generate test cases containing:
Test Case ID
Scenario
Test Steps
Test Data
Expected Result
Priority 
If test type is negative use temperature as 0.5 and 
if test type is positive use temperature as 0 
and if test type is all use temperature as 0.7."""

user_prompt = f""" These are the user requirement : 
                Requirement :{userReqInput}
                Test Type : {testType}
                Number of Test Cases : {noOfTestCases}
Please generate test cases based on the above user requirement and test type."""

load_dotenv()
agent = create_agent(model="groq:openai/gpt-oss-20b", system_prompt=System_prompt)
result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

print(result["messages"][-1].content)
