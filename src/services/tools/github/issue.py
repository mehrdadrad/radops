import logging
from typing import Annotated, List, Optional

from github.GithubObject import NotSet
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from .helper import get_github_repo

logger = logging.getLogger(__name__)

class GitHubBaseInput(BaseModel):
    """Base input schema for GitHub tools."""
    user_id: Annotated[str, InjectedState("user_id")] = Field(
        description="The user's ID to retrieve credentials for GitHub."
    )

class ListIssuesInput(GitHubBaseInput):
    """Input for the github_list_issues tool."""
    assignee: Optional[str] = Field(default=None, description="The GitHub username the issue is assigned to.")
    labels: Optional[List[str]] = Field(default=None, description="A list of labels to filter by (e.g., ['bug', 'documentation']).")
    state: str = Field(default='open', description="The state of the issues to find ('open', 'closed', 'all').")

class CreateIssueInput(GitHubBaseInput):
    """Input for the github_create_issue tool."""
    title: str = Field(..., description="The title of the new issue.")
    body: str = Field(..., description="The detailed description of the issue.")
    assignee: Optional[str] = Field(default=None, description="The GitHub username to assign the issue to.")
    labels: Optional[List[str]] = Field(default=None, description="A list of labels to add to the new issue.")

@tool(args_schema=ListIssuesInput)
def github_list_issues(
    user_id: Annotated[str, InjectedState("user_id")],
    assignee: Optional[str] = None,
    labels: Optional[List[str]] = None,
    state: str = 'open'
) -> list:
    """
    Finds open bugs or tasks assigned to the agent.

    Returns:
        A list of dictionaries, each representing an issue.
    """
    repo, error = get_github_repo(user_id)
    if not repo:
        return [error]
        
    try:
        kwargs = {'state': state, 'labels': labels or []}
        if assignee:
            kwargs['assignee'] = assignee

        issues = repo.get_issues(**kwargs)
        
        issue_list = []
        for issue in issues:
            issue_list.append({
                "number": issue.number,
                "title": issue.title,
                "url": issue.html_url,
                "labels": [label.name for label in issue.labels]
            })
        return issue_list
    except Exception as e:
        logger.error(f"Error in github_list_issues: {e}")
        return [{"error": str(e)}]

@tool(args_schema=CreateIssueInput)
def github_create_issue(
    user_id: Annotated[str, InjectedState("user_id")],
    title: str,
    body: str,
    assignee: Optional[str] = None,
    labels: Optional[List[str]] = None
) -> dict:
    """
    Allows the agent to report bugs or break down a large task into smaller tickets.

    Returns:
        A dictionary containing the new issue's number and URL, or an error message.
    """
    repo, error = get_github_repo(user_id)
    if not repo:
        return error
    
    if not assignee:
        assignee = NotSet

    try:
        issue = repo.create_issue(
            title=title,
            body=body,
            assignee=assignee,
            labels=labels or []
        )
        return {
            "number": issue.number,
            "url": issue.html_url,
            "message": f"Successfully created issue #{issue.number}."
        }
    except Exception as e:
        logger.error(f"Error in github_create_issue: {e}")
        return {"error": repr(e)}
