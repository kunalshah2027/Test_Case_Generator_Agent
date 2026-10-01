import streamlit as st
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import BaseModel

System_prompt = """You are a Senior QA Engineer person."""

user_prompt = """create 1 test cases for login page."""


class test_case_format(BaseModel):
    title: str
    description: str
    steps: str
    expected_result: str
    test_data: str


load_dotenv()
agent = create_agent(model="groq:openai/gpt-oss-20b", system_prompt=System_prompt,
                     response_format=test_case_format)

result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
print(result["structured_response"])
