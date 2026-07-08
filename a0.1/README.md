# Nexus_Assignments
This repo is provided to be a host of Nexus course assignments.

# Homework Submission Instructions

This document outlines the steps to clone, complete, and submit homework assignments using the provided `utils.py` script.

## Prerequisites

- Ensure you have Python installed on your system.
- Install Git on your machine.
- Set up SSH keys for secure Git operations.

## Setup Instructions

1. **Configure SSH Keys**:

   - Generate an SSH key pair if you don't already have one:

     ```bash
     ssh-keygen -t ed25519 -C "your_email@example.com"
     ```

   - Add the public key to your Git hosting service (e.g., GitHub, GitLab).

   - Start the SSH agent and add your private key:

     ```bash
     eval "$(ssh-agent -s)"
     ssh-add ~/.ssh/id_ed25519
     ```

## Cloning and Submitting Homework

Follow these steps for each homework assignment (e.g., Homework 0.3):

1. **Update Local Repository**:

   - Before starting, ensure your local repository is up-to-date:

     ```bash
     git pull
     ```

2. **Clone the Homework**:

   - Use the `utils.py` script to clone the homework assignment:

     ```bash
     python utils.py <homework_number>
     ```

     Replace `<homework_number>` with the specific homework number (e.g., `0.3`).

3. **Complete the Homework**:

   - Navigate to the cloned homework directory.
   - Complete the required tasks or assignments as per the instructions.

4. **Commit and Push Changes**:

   - Stage your changes:

     ```bash
     git add .
     ```

   - Commit your changes with a descriptive message:

     ```bash
     git commit -m "Completed Homework <homework_number>"
     ```

   - Push your changes to the remote repository:

     ```bash
     git push origin main
     ```

## Important Notes

- Always run `git pull` before starting work to avoid conflicts and ensure you have the latest updates.
- Follow the clone*-complete-push procedure for every homework assignment.
- *: here clone means using utils.py command to clone the homework not git clone.

Verify your SSH keys are correctly configured to avoid authentication issues.
