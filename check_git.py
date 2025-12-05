import subprocess
import os

def run_git_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return f"Command: {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n"
    except Exception as e:
        return f"Command: {cmd}\nFailed: {e}\n"

output = ""
output += run_git_cmd("git status")
output += run_git_cmd("git remote -v")

with open("git_info.txt", "w") as f:
    f.write(output)
