from __future__ import annotations

from github import Github


class GitHubService:
    def __init__(self, token: str | None):
        self.token = token

    def _client(self) -> Github:
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN is not configured.")
        return Github(self.token)

    def list_repos(self, limit: int = 20) -> list[dict]:
        user = self._client().get_user()
        repos = []
        for repo in user.get_repos(sort="updated"):
            repos.append({"name": repo.full_name, "private": repo.private, "url": repo.html_url, "description": repo.description})
            if len(repos) >= limit:
                break
        return repos

    def list_issues(self, owner: str, repo: str, state: str = "open", limit: int = 20) -> list[dict]:
        issues = self._client().get_repo(f"{owner}/{repo}").get_issues(state=state)
        output = []
        for issue in issues:
            output.append({"number": issue.number, "title": issue.title, "state": issue.state, "url": issue.html_url})
            if len(output) >= limit:
                break
        return output

    def create_issue(self, owner: str, repo: str, title: str, body: str = "") -> dict:
        issue = self._client().get_repo(f"{owner}/{repo}").create_issue(title=title, body=body)
        return {"created": True, "number": issue.number, "title": issue.title, "url": issue.html_url}
