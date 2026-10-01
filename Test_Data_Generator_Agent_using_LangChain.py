from typing import List

import streamlit as st
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import BaseModel, Field

scenario = input("Please enter your scenario: ")


class rsp_format(BaseModel):
    category: str = Field(
        description="The category of the test data. Must be either 'Valid', 'Invalid', or 'Boundary'."
    )
    input_value: str = Field(
        description="The specific Test Data string/value generated for this test case based on the scenario."
    )
    expected_result: str = Field(
        description="Whether the application should 'accepted' or 'rejected' this input_value."
    )
    reason: str =  Field(
        description="Detailed reason explaining why this specific input is accepted or rejected."
    )


class TestDataResponse(BaseModel):
    test_data: List[rsp_format] = Field(
        description="A comprehensive list containing multiple test cases. Generates at least one valid, one invalid, "
                    "and one boundary condition case."
    )


# System_prompt = f"""You are a Senior QA Engineer person Your responsibility is to analyse the requirement given by user
# and generate output in structured json format mentioned in {rsp_format} class"""

System_prompt = """You are a Senior QA Engineer person Your responsibility is to accept test scenario from the user 
and generate structured test data.
Requirements:
- You must generate multiple test cases.
- Provide a diverse set of test data including Valid Test Data, Invalid Test Data, and Boundary Test Data.
- Return the full list wrapped inside the 'test_cases' structure."""

user_prompt = f""" This is the user's scenario : 
                Scenario :{scenario}
Please analyze scenario and generate Test data list"""

load_dotenv()
agent = create_agent(model="groq:openai/gpt-oss-20b", system_prompt=System_prompt,
                     response_format=TestDataResponse)

result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
print(type(result["structured_response"]))
# print(result["structured_response"])
pydantic_obj = result["structured_response"]
json_string = pydantic_obj.model_dump_json(indent=4)
print(json_string)
