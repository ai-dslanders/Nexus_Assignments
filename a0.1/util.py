import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import argparse

env_dict = {
    "FOLDER_TEMPLATE": "a{version}",
    "FILE_TEMPLATE": "AI-DS_Nexus__A{version_with_underscore}",
    "FINAL_LITERAL": "__RezaShokr"
}

def get_version(version=None):
    if version is None:
        version = input("Version: ")
    return version, version.replace('.', '_')

def get_username():
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            check=True
        )
        raw_username = result.stdout.strip()
        sanitized_username = re.sub(r'\W+', '_', raw_username)
        return sanitized_username
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to retrieve git username: {e}")

def create_or_switch_branch(branch_name):
    try:
        result = subprocess.run(
            ["git", "branch", "--list"],
            capture_output=True,
            text=True,
            check=True
        )
        branches = [line.strip().lstrip("* ") for line in result.stdout.splitlines()]

        if branch_name in branches:
            print(f"Branch {branch_name} is exists. only switching.")
            subprocess.run(["git", "switch", branch_name], check=True)
        else:
            subprocess.run(["git", "switch", "-c", branch_name], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create or switch branch: {e}")

def commit_changes(files_count, files_name, version, base_message="Add user specific files for version {}: {}."):
    try:
        if files_count > 0:
            subprocess.run(["git", "add", "-A"], check=True, text=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", base_message.format(version, files_name)],
                check=True,
                text=True,
                capture_output=True
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to commit changes: {e}")

def get_hash(value, length=6):
    return hashlib.sha256(value.encode()).hexdigest()[:length]

def create_user_copies(folder_template, file_template, folder_version, file_version, username, final_literal):
    folder = Path(folder_template.format(version=folder_version))
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Homework not found: {folder_version}")

    file_basepattern = file_template.format(version_with_underscore=file_version)
    hash_val = get_hash(username)

    files_counter, files_name = 0, []
    for file in folder.iterdir():
        if file.is_file() and file.stem.startswith(file_basepattern) and file.stem.endswith(final_literal):
            base = file.stem.rsplit(final_literal, 1)[0]
            new_stem = f"{base}_{username}_{hash_val}"
            new_file = file.with_stem(new_stem)
            if not new_file.exists():
                shutil.copy(file, new_file)
                print(f"File {file.name} copied to {new_file.name}")
                files_name.append(new_file.name)
                files_counter += 1
            else:
                print(f"File {new_file.name} already exists, skipping copy.")
    return files_counter, files_name

def main():
    parser = argparse.ArgumentParser(
        description="This script creates a Git branch based on your username, copies version-specific files from a template folder, appends your sanitized username and a hash to the filenames, and commits the changes if any new files are created."
    )
    parser.add_argument(
        'version',
        nargs='?',
        help="The version number (e.g., '1.0'). If not provided, the script will prompt you to enter it."
    )

    args = parser.parse_args()

    folder_template = env_dict["FOLDER_TEMPLATE"]
    file_template = env_dict["FILE_TEMPLATE"]
    final_literal = env_dict["FINAL_LITERAL"]

    folder_version, file_version = get_version(args.version)
    username = get_username()

    create_or_switch_branch(username)

    files_count, files_name = create_user_copies(folder_template, file_template, folder_version, file_version, username, final_literal)

    commit_changes(files_count, files_name, folder_version)

if __name__ == '__main__':
    main()
