from typing import List

import streamlit as st
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import BaseModel, Field

userReqInput = input("Please enter your requirement: ")

# userReqInput = """A user should be able to transfer money from their bank account
# to another users account. The user must enter the beneficiary
# account number and transfer amount. The minimum transfer amount
# is ₹100 and the maximum amount is ₹50,000 per transaction.
# The system should not allow the transfer if the account balance
# is insufficient. After a successful transfer, the user should
# receive a confirmation message."""


class rsp_format(BaseModel):

    main_features: List[str] = Field(
        description="Core functionalities extracted from the user requirements."
    )
    acceptance_criteria: List[str] = Field(
        description="Conditions that the software must satisfy to be accepted by the user."
    )
    missing_information: List[str] = Field(
        description="Ambiguities, gaps, or questions that need clarification from the user."
    )
    testing_risks: List[str] = Field(
        description="Potential challenges, edge cases, or risks during the testing phase."
    )


# System_prompt = f"""You are a Senior QA Engineer person Your responsibility is to analyse the requirement given by user
# and generate output in structured json format mentioned in {rsp_format} class"""

System_prompt = """You are a Senior QA Engineer person Your responsibility is \
to analyse the requirement given by user"""

user_prompt = f""" These are the user requirement : 
                Requirement :{userReqInput}
Please analyze this user requirement thoroughly."""

load_dotenv()
agent = create_agent(model="groq:openai/gpt-oss-20b", system_prompt=System_prompt,
                     response_format=rsp_format)

result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
print(type(result["structured_response"]))
#print(result["structured_response"])
pydantic_obj = result["structured_response"]
json_string = pydantic_obj.model_dump_json(indent=4)
print(json_string)