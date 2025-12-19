# pull_request.py
from typing import Annotated

from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from github import GithubException
from .helper import get_github_repo

import logging

logger = logging.getLogger(__name__)

class GitHubBaseInput(BaseModel):
    """Base input schema for GitHub tools."""
    user_id: Annotated[str, InjectedState("user_id")] = Field(
        description="The user's ID to retrieve credentials for GitHub."
    )

class CreatePullRequestInput(GitHubBaseInput):
    """Input for the github_create_pull_request tool."""
    title: str = Field(..., description="The title of the new pull request.")
    body: str = Field(..., description="The contents of the pull request.")
    head: str = Field(..., description="The name of the branch where your changes are implemented. For cross-repository pull requests in the same network, namespace head with a user like: username:branch.")
    base: str = Field(..., description="The name of the branch you want the changes pulled into. This should be an existing branch on the current repository.")
    draft: bool = Field(default=False, description="Indicates whether the pull request is a draft.")

class ListPullRequestsInput(GitHubBaseInput):
    """Input for the github_list_pull_requests tool."""
    state: str = Field(default='open', description="The state of the pull requests to find ('open', 'closed', 'all').")
    sort: str = Field(default='created', description="What to sort results by. Can be either 'created', 'updated', 'popularity', 'long-running'.")
    direction: str = Field(default='desc', description="The direction of the sort. Can be either 'asc' or 'desc'.")

@tool(args_schema=CreatePullRequestInput)
def github_create_pull_request(user_id: Annotated[str, InjectedState("user_id")], title: str, body: str, head: str, base: str, draft: bool = False) -> dict:
    """
    Create a new pull request.

    Returns:
        A dictionary containing the new pull request's number and URL, or an error message.
    """
    repo, error = get_github_repo(user_id)
    if not repo:
        return error
    
    try:
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head,
            base=base,
            draft=draft
        )
        return {
            "number": pr.number,
            "url": pr.html_url,
            "message": f"Successfully created pull request #{pr.number}."
        }
    except GithubException as e:
        error_message = e.data.get("message", "An unknown validation error occurred.")
        if e.status == 422 and "errors" in e.data:
            for err in e.data["errors"]:
                if err.get("resource") == "PullRequest":
                    if err.get("field") == "base" and err.get("code") == "invalid":
                        error_message = f"Failed to create pull request. The base branch '{base}' is invalid or does not exist. Please specify an existing branch."
                        break
                    elif "A pull request already exists" in err.get("message", ""):
                        error_message = f"Failed to create pull request. A pull request for '{head}' into '{base}' already exists."
                        break
        logger.error(f"GitHub API error in github_create_pull_request: {e}")
        return {"error": error_message}
    except Exception as e:
        logger.error(f"Error in github_create_pull_request: {e}")
        return {"error": repr(e)}

@tool(args_schema=ListPullRequestsInput)
def github_list_pull_requests(user_id: Annotated[str, InjectedState("user_id")], state: str = 'open', sort: str = 'created', direction: str = 'desc') -> list:
    """
    List pull requests in the repository.

    Returns:
        A list of dictionaries, each representing a pull request.
    """
    repo, error = get_github_repo(user_id)
    if not repo:
        return [error]
        
    try:
        pulls = repo.get_pulls(state=state, sort=sort, direction=direction)
        
        pr_list = []
        for pr in pulls:
            pr_list.append({
                "number": pr.number,
                "title": pr.title,
                "url": pr.html_url,
                "state": pr.state,
                "user": pr.user.login,
                "created_at": str(pr.created_at)
            })
        return pr_list
    except GithubException as e:
        logger.error(f"Error in github_list_pull_requests: {e}")
        return [{"error": f"GitHub API error: {e.data.get('message', 'An unknown error occurred.')}"}]
    except Exception as e:
        logger.error(f"Error in github_list_pull_requests: {e}")
        return [{"error": str(e)}]