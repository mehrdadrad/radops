from utils.secrets import get_user_secrets
from config.tools import tool_settings as settings
from github import Github, GithubException, Auth

def get_github_repo(user_id: str, org: str = None, repo_name: str = None):
    """Helper function to get authenticated GitHub repository object."""
    try:
        credentials = get_user_secrets(user_id=user_id, service='github', raise_on_error=True)

        github_token = credentials.get('token')
        github_org = org or credentials.get('github_org', settings.github.default_org)
        github_repo = repo_name or credentials.get('github_repo', settings.github.default_repo)

        if not all([github_token, github_org, github_repo]):
            return None, {"error": "GitHub token, repository owner, or repository name not found in secrets."}

        auth = Auth.Token(github_token)
        if settings.github.server and len(settings.github.server) > 0:
            g = Github(base_url=settings.github.server, auth=auth)
        else:
            g = Github(auth=auth)
        
        repo = g.get_repo(f"{github_org}/{github_repo}")
        return repo, None
    except GithubException as e:
        if e.status == 404:
            return None, {"error": f"Repository '{github_org}/{github_repo}' not found or token has insufficient permissions."}
        return None, {"error": f"GitHub API error: {e}"}
    except Exception as e:
        return None, {"error": f"Failed to initialize GitHub repository: {e}"}