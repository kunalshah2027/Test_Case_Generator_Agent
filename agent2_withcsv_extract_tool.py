import json
import re

import pandas as pd
import streamlit as st
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.tools import tool

st.title('Test Case Generator')

# Initialize session state variables to persist data across clicks
if "df_data" not in st.session_state:
    st.session_state.df_data = None
if "csv_bytes" not in st.session_state:
    st.session_state.csv_bytes = None

userReqInput = st.text_area(label='Requirement', placeholder='Please enter your requirement')
testType = st.selectbox('Please enter test Type', ['positive', 'negative', 'all'])
noOfTestCases = st.text_input('Please enter number of test cases')

System_prompt = """You are a Senior QA Engineer person.Your responsibility is to analyse teh requirement and generate
 test cases based on the  user input.

You MUST respond strictly with a valid JSON array containing objects for each test case. Do not wrap the JSON in 
markdown code blocks like ```json. Do not include any conversational text before or after the JSON.

Generate test cases and Each object in the JSON array must contain exactly these keys:
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
Please generate exactly {noOfTestCases} test cases based on the requirements in the requested JSON array format."""

load_dotenv()
agent = create_agent(model="groq:openai/gpt-oss-20b", system_prompt=System_prompt)


def convert_to_csv(raw_llm_output):
    try:
        # Strip deepseek/groq reasoning <think> tags if present
        clean_text = re.sub(r'<think>.*?</think>', '', raw_llm_output, flags=re.DOTALL).strip()

        # STRATEGY 1: Parse Tab-Separated Tables (The current model layout)
        if "Test Case ID" in clean_text and "\t" in clean_text:
            lines = clean_text.split('\n')
            table_lines = []
            start_collecting = False

            for line in lines:
                # Find the main table header row
                if "Test Case ID" in line and "Scenario" in line:
                    start_collecting = True
                if start_collecting:
                    if line.strip() and ("TC" in line or "Test Case ID" in line):
                        table_lines.append(line)
                    elif not line.strip() and len(table_lines) > 1:
                        # Stop if we hit an empty line after collecting table rows
                        break

            if len(table_lines) >= 2:
                table_str = '\n'.join(table_lines)
                df = pd.read_csv(io.StringIO(table_str), sep='\t')
                # Clean clean HTML breaking elements from model
                df = df.replace(r'<br\s*/?>', '\n', regex=True)
                return df.to_csv(index=False).encode('utf-8'), df

        # STRATEGY 2: Parse standard Pipeline Markdown tables (| ID | Scenario |)
        if "|" in clean_text:
            lines = [line.strip() for line in clean_text.split('\n') if '|' in line]
            if len(lines) >= 2:
                table_str = '\n'.join(lines)
                df = pd.read_csv(io.StringIO(table_str), sep=r'\s*\|\s*', engine='python')
                df = df.dropna(how='all', axis=1)
                df.columns = [col.strip() for col in df.columns]
                df = df[~df.iloc[:, 0].astype(str).str.contains(r'^[-:]+$')]
                df = df.replace(r'<br\s*/?>', '\n', regex=True)
                return df.to_csv(index=False).encode('utf-8'), df

        # STRATEGY 3: Parse blocks formatted with direct text names ("Test Case ID: TC001")
        if "Test Case ID:" in clean_text or "Scenario:" in clean_text:
            blocks = re.split(r'(?:Test Case\s+\d+[:\-]*|\bTest Case ID:)', clean_text)
            test_cases = []
            for block in blocks:
                if not block.strip(): continue
                tc_id = re.search(r'(?:Test Case ID:\s*)(.*?)(?=\n|$|\bScenario:)', block, re.IGNORECASE)
                scenario = re.search(r'(?:Scenario:\s*)(.*?)(?=\n|$|\bTest Steps:)', block, re.IGNORECASE)
                steps = re.search(r'(?:Test Steps:\s*)(.*?)(?=\n|$|\bTest Data:)', block, re.DOTALL | re.IGNORECASE)
                data = re.search(r'(?:Test Data:\s*)(.*?)(?=\n|$|\bExpected Result:)', block, re.DOTALL | re.IGNORECASE)
                expected = re.search(r'(?:Expected Result:\s*)(.*?)(?=\n|$|\bPriority:)', block,
                                     re.DOTALL | re.IGNORECASE)
                priority = re.search(r'(?:Priority:\s*)(.*?)(?=\n|$|\bTemperature:|\bTest Case|$)', block,
                                     re.IGNORECASE)

                if tc_id or scenario:
                    clean_steps = steps.group(1).strip() if steps else ""
                    clean_steps = re.sub(r'<br\s*/?>', '\n', clean_steps)
                    clean_data = data.group(1).strip() if data else ""
                    clean_data = re.sub(r'<br\s*/?>', '\n', clean_data)

                    test_cases.append({
                        "Test Case ID": tc_id.group(1).strip() if tc_id else f"TC_{len(test_cases) + 1}",
                        "Scenario": scenario.group(1).strip() if scenario else "",
                        "Test Steps": clean_steps,
                        "Test Data": clean_data,
                        "Expected Result": expected.group(1).strip() if expected else "",
                        "Priority": priority.group(1).strip() if priority else "Medium"
                    })
            if test_cases:
                df = pd.DataFrame(test_cases)
                return df.to_csv(index=False).encode('utf-8'), df

        # STRATEGY 4: Standard JSON array backup rule
        json_match = re.search(r'\[\s*\{.*\}\s*\]', clean_text, flags=re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)
        else:
            clean_json_str = re.sub(r'^```json\s*|```$', '', clean_text, flags=re.MULTILINE).strip()

        data = json.loads(clean_json_str)
        df = pd.DataFrame(data)
        df = df.replace(r'<br\s*/?>', '\n', regex=True)
        return df.to_csv(index=False).encode('utf-8'), df

    except Exception as e:
        st.error(f"Failed to parse test cases into CSV. Error: {e}")
        st.info(f"Raw Output Received:\n{raw_llm_output}")
        return None, None


if st.button('Generate Test Cases'):
    with st.spinner("Generating test cases ..."):
        result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        raw_output = result["messages"][-1].content
        # st.write(result["messages"][-1].content)

        # Process CSV and DataFrame
        csv_bytes, df = convert_to_csv(raw_output)

        if df is not None:
            st.session_state.df_data = df
            st.session_state.csv_bytes = csv_bytes
            st.success("Test cases generated successfully!")

            # Show an interactive data table in UI

if st.session_state.df_data is not None:
    st.markdown("---")  # Visual separator
    st.subheader("Generated Test Cases Preview")
    st.dataframe(st.session_state.df_data, use_container_width=True)

    # Provide a stable Download Button for the CSV
    st.download_button(
        label="📥 Download Test Cases as CSV",
        data=st.session_state.csv_bytes,
        file_name=f"test_cases_{testType}.csv",
        mime="text/csv",
        key="download_csv_button"
    )
