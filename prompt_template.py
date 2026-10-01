import streamlit as st
from langchain.agents import create_agent
from dotenv import load_dotenv

System_prompt = """You are a Senior QA Engineer person."""

user_prompt = """create 2 test cases for login page."""

load_dotenv()
agent = create_agent(model="groq:openai/gpt-oss-20b", system_prompt=System_prompt)

result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
print(result["messages"][-1].content)
