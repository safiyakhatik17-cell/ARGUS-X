"""
GitHub Upload Utility
Creates public repository 'TRACK-THE-UNTRACKABLE' and uploads clean project files via GitHub REST API.
"""

import os
import sys
import base64
import requests
from typing import List, Dict, Any


def upload_repository(token: str, repo_name: str = "TRACK-THE-UNTRACKABLE") -> str:
    """
    Creates and pushes files to GitHub repository using GitHub REST API.
    """
    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # 1. Get authenticated user
    user_res = requests.get("https://api.github.com/user", headers=headers)
    if user_res.status_code != 200:
        raise RuntimeError(f"GitHub authentication failed ({user_res.status_code}): {user_res.text}")

    username = user_res.json().get("login")
    print(f"Authenticated as GitHub user: {username}")

    # 2. Check if repository already exists or create it
    repo_res = requests.get(f"https://api.github.com/repos/{username}/{repo_name}", headers=headers)
    if repo_res.status_code == 404:
        print(f"Creating public repository '{repo_name}'...")
        create_payload = {
            "name": repo_name,
            "description": "Long-Horizon Multi-Object Identity Recovery Under Camera, Occlusion & Appearance Drift",
            "private": False,
            "auto_init": False
        }
        create_res = requests.post("https://api.github.com/user/repos", headers=headers, json=create_payload)
        if create_res.status_code not in (201, 200):
            raise RuntimeError(f"Failed to create repo ({create_res.status_code}): {create_res.text}")
        print("Repository created successfully!")
    elif repo_res.status_code == 200:
        print(f"Repository '{repo_name}' already exists.")
    else:
        raise RuntimeError(f"Error checking repo ({repo_res.status_code}): {repo_res.text}")

    # 3. Collect files to upload
    exclude_exts = {'.pt', '.pth', '.mp4', '.pyc', '.zip'}
    exclude_dirs = {'__pycache__', '.git', '.cache', 'output'}

    files_to_upload: List[str] = []
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in exclude_exts or f == "upload_to_github.py":
                continue
            full_path = os.path.join(root, f)
            arcname = os.path.relpath(full_path, '.').replace('\\', '/')
            files_to_upload.append(arcname)

    print(f"Uploading {len(files_to_upload)} project files to GitHub...")

    for file_path in files_to_upload:
        with open(file_path, "rb") as f:
            content_bytes = f.read()

        b64_content = base64.b64encode(content_bytes).decode('utf-8')
        api_url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{file_path}"

        # Check if file exists to retrieve SHA for update
        sha = None
        get_file_res = requests.get(api_url, headers=headers)
        if get_file_res.status_code == 200:
            sha = get_file_res.json().get("sha")

        commit_data = {
            "message": f"Add {file_path}",
            "content": b64_content
        }
        if sha:
            commit_data["sha"] = sha

        put_res = requests.put(api_url, headers=headers, json=commit_data)
        if put_res.status_code in (200, 201):
            print(f"  [OK] Uploaded: {file_path}")
        else:
            print(f"  [WARN] Failed to upload {file_path}: {put_res.text}")

    repo_url = f"https://github.com/{username}/{repo_name}"
    print(f"\nSUCCESS! Public repository live at: {repo_url}")
    return repo_url


if __name__ == "__main__":
    token = os.environ.get("GITHUB_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]

    if not token:
        print("ERROR: Please provide a GitHub Personal Access Token.")
        sys.exit(1)

    url = upload_repository(token)
    print("REPO_URL=" + url)
