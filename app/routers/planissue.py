"""
Issue planning router module.

This module provides endpoints for AI-powered issue planning functionality.
It allows users to submit issue details and receive AI-generated task breakdowns
with estimated time requirements and test cases in Gherkin format.

The module integrates with AI providers through the service layer to analyze
issue requirements and generate comprehensive implementation plans including
task dependencies, time estimates, and test scenarios.

Author:
    Rana Nouman <ranamnouman@gmail.com>
"""

from fastapi import APIRouter, Query
from app.services.planissue_service import get_ai_response
import re

router = APIRouter(
    prefix="/planissue",
    tags=["planissue"]
)

@router.post("/")
async def plan_issue_with_ai(issue_details: dict):
    """
    Plan issues using AI providers to generate task breakdowns and test cases.
    
    This endpoint accepts issue details and uses AI to generate a comprehensive
    implementation plan. The AI analyzes the issue in the context of the project
    and breaks it down into manageable tasks with dependencies, time estimates,
    and Gherkin-format test cases.
    
    The endpoint supports different project contexts (Prometheus and Gaia) and
    generates structured JSON output containing tasks, dependencies, and test
    scenarios that can be directly used for project planning and implementation.
    
    Args:
        issue_details (dict): A dictionary containing issue information with
            the following structure:
            {
                "issueDetails": {
                    "projectName": str,      # Name of the project
                    "issueDescription": str, # Detailed issue description
                    "issueSubject": str      # Issue subject/title
                }
            }
    
    Returns:
        str: A JSON string containing the AI-generated task breakdown with
            the following structure:
            {
                "tasks": [
                    {
                        "id": str,              # Unique task identifier
                        "subject": str,         # Task title
                        "description": str,     # Detailed task description
                        "dependency": str,      # ID of dependent task (or null)
                        "estimated_hours": int, # Time estimate in hours
                        "tests": [              # List of test cases
                            {
                                "scenario": str,      # Test scenario name
                                "test_case": str,     # Gherkin test case
                                "estimated_hours": int # Test implementation time
                            }
                        ]
                    }
                ]
            }
    
    Raises:
        ValueError: If the AI response doesn't contain a valid JSON block
            or if the response format is unexpected.
        Exception: For other errors during AI processing or response parsing.
    
    Example:
        POST /planissue/
        Request Body:
        {
            "issueDetails": {
                "projectName": "A Prometheus",
                "issueDescription": "Add user authentication feature",
                "issueSubject": "Implement login functionality"
            }
        }
        
        Response:
        {
            "tasks": [
                {
                    "id": "TASK-1",
                    "subject": "Create login form component",
                    "description": "Implement the login form UI...",
                    "dependency": null,
                    "estimated_hours": 4,
                    "tests": [...]
                }
            ]
        }
    """
    provider = "anthropic"
    issue_details = issue_details.get("issueDetails")
    project_name = issue_details.get("projectName")
    issue_description = issue_details.get("issueDescription")
    issue_subject = issue_details.get("issueSubject")
    project_description = ""

    if(project_name == "A Prometheus"):
        project_description = "Prometheus is the graphic user interface of Projects4Me. It is built on top of EmberJS and Bootstrap 3."
    elif(project_name == "A Gaia"):
        project_description = "Gaia is the core of the Projects4Me. Gaia maintains all the data and behavior of the application. Gaia is implemented on top of PhalconPHP."

    user_prompt = f"""The description of our project in which I'm working on is {project_description}.
                    Having the following details of the issue embedded in JSON, Enchance the functional requirements and give me a list of tasks that will need to be implemented to complete this functionality. Also give me estimated time for each task. Also add Test cases in Gherkin format with estimations. Embed everything inside the JSON.

                    Issue details: (Input)
                    [
                        {{
                            "field": "subject",
                            "detail": "The subject containing the outline of the issue",
                            "value": "{issue_subject}"
                        }},
                        {{
                            "field": "description",
                            "detail": "Issue description. This contains the detail of the issue",
                            "value": "{issue_description}"
                        }}
                    ]

                    **Your task:**  
                    Analyze the above issue in the context of the project and perform the following:

                    1. Break down the functionality into **a list of tasks** required to fully implement the described issue.
                        - Each task should include:
                            - `"id"`: Unique identifier for the task
                            - `"subject"`: A concise title of the task
                            - `"description"`: A detailed explanation of what needs to be done
                            - `"dependency"`: The `id` of any task it depends on (or `null` if it's standalone)
                            - `"estimated_hours"`: Estimated time to complete the task
                            - `"tests"`: A list of related test cases (see format below)

                    2. Each task must include a `"tests"` key which contains the Gherkin-format test cases required to verify that task. Each test case should have:
                        - `"scenario"`: High-level name of the scenario
                        - `"test_case"`: Gherkin-style test case
                        - `"estimated_hours"`: Time to implement and validate the test

                    Output everything as a **single JSON object** with a top-level `"tasks"` array. Each item in this array must include its associated `"tests"` inline.
                    """

    assistant_prompt = """
                    1. Add a json placeholder e.g. "```json" for the output in the response that indicates the start of the output.
                    2. Issues should be break down into smaller tasks that can be doable within 8 hours, if more than 8 hours, break down into multiple tasks.
                    3. Add a json placeholder e.g. "```" for the end of the output.
                    4. Don't a separate section ahead of the tests section for the ambigious tasks.
                    5. The tasks should be in the order of the dependencies.
                    6. The tasks should be clear enough to be implemented.
                    Follow this structure strictly in your response:

                    ```json
                    {
                    "tasks": [
                        {
                        "id": "TASK-1",
                        "subject": "Short task title",
                        "description": "Detailed explanation of the task",
                        "dependency": null,
                        "estimated_hours": 4,
                        "tests": [
                            {
                            "scenario": "Short scenario name",
                            "test_case": "Gherkin-formatted test case",
                            "estimated_hours": 1
                            }
                        ]
                        },
                        {
                        "id": "TASK-2",
                        "subject": "Dependent task title",
                        "description": "Another detailed explanation",
                        "dependency": "TASK-1",
                        "estimated_hours": 3,
                        "tests": [
                            {
                            "scenario": "Another scenario",
                            "test_case": "Gherkin-formatted test case",
                            "estimated_hours": 1
                            }
                        ]
                        }
                    ]
                    }"""
    message = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt}
    ]

    response = await get_ai_response(message, provider) 
    match = re.search(r'```json\n(.*?)```', response['response'], re.DOTALL)
    if not match:
        raise ValueError("JSON block not found in the response.")

    json_string = match.group(1)
    return json_string