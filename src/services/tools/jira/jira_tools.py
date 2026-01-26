from typing import Annotated

from langgraph.prebuilt import InjectedState
from jira import JIRA
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field

from config.tools import tool_settings as settings
from utils.secrets import get_user_secrets

class JiraBaseInput(BaseModel):
    """Base input schema for Jira tools."""
    user_id: Annotated[str, InjectedState("user_id")] = Field(
        description="The user's ID to retrieve credentials for Jira."
    )


class CreateJiraTicketInput(JiraBaseInput):
    """Input for the create_jira_ticket tool."""
    project: str = Field(
        ...,
        description="The Jira project key (e.g., 'PROJ'), this is required."
    )
    summary: str = Field(
        ...,
        description="The summary or title of the ticket, this is required."
    )
    description: str = Field(
        ...,
        description="The detailed description for the ticket, this is required."
    )
    issue_type: str = Field(
        default="Task",
        description="The type of the issue (e.g., 'Task', 'Bug', 'Story')."
    )


class SearchJiraIssuesInput(JiraBaseInput):
    """Input for the search_jira_issues tool."""
    jql_query: str = Field(
        ...,
        description=(
            "The JQL query string to use for searching "
            "(e.g., 'project = \"PROJ\" AND status = \"In Progress\"')."
        )
    )
    max_results: int = Field(
        default=20,
        description="The maximum number of issues to return."
    )


@tool(args_schema=CreateJiraTicketInput)
async def jira_create_ticket(
    user_id: Annotated[str, InjectedState("user_id")],
    project: str,
    summary: str,
    description: str,
    issue_type: str = "Task"
) -> str:
    """
    Use this tool to create a new issue in Jira when a user wants to create
    a task, bug, or story.
    """
    try:
        credentials = get_user_secrets(
            user_id=user_id, service='jira', raise_on_error=True
        )
        if 'error' in credentials:
            raise ValueError(
                f"error retrieving Jira credentials ({credentials['error']})."
            )

        username = credentials.get('username')
        api_key = credentials.get('token')

        jira_options = {'server': settings.jira.server}
        jira = JIRA(options=jira_options, basic_auth=(username, api_key))

        # Validate project existence
        jira.project(project)

        issue_dict = {
            'project': {'key': project},
            'summary': summary,
            'description': description,
            'issuetype': {'name': issue_type},
        }
        new_issue = jira.create_issue(fields=issue_dict)
        return f"Successfully created Jira ticket: {new_issue.key}"
    except Exception as e:
        raise ToolException(f"Error creating Jira ticket: {e}")


@tool(args_schema=SearchJiraIssuesInput)
async def jira_search_issues(
    user_id: Annotated[str, InjectedState("user_id")],
    jql_query: str,
    max_results: int = 20
) -> str:
    """
    Use this tool to search for issues in Jira using a JQL query.
    Before calling this tool, you must call 'get_app_token' for the 'jira'
    service to get the username and API key.
    """
    try:
        credentials = get_user_secrets(
            user_id=user_id, service='jira', raise_on_error=True
        )
        if 'error' in credentials:
            raise ValueError(
                f"Error retrieving Jira credentials ({credentials['error']})."
            )

        username = credentials.get('username')
        api_key = credentials.get('token')

        jira_options = {'server': settings.jira.server}
        jira = JIRA(options=jira_options, basic_auth=(username, api_key))

        issues = jira.search_issues(jql_query, maxResults=max_results)

        if not issues:
            return "No issues found matching the query."

        results = [
            f"- {issue.key}: {issue.fields.description} "
            f"(Status: {issue.fields.status.name})"
            for issue in issues
        ]

        response = f"Found {len(issues)} issues:\n" + "\n".join(results)
        if len(issues) == max_results:
            response += (
                f"\nNote: The number of results is limited to {max_results}. "
                "There might be more issues matching your query."
            )
        return response
    except Exception as e:
        raise ToolException(f"Error searching Jira issues: {e}")