import subprocess
import sys
import time
import re
import yaml
from pathlib import Path

_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yml"))["download"]


def query_ollama_model(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", _cfg["ollama_model"], "--hidethinking"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return None


def get_video_theme_prompt(theme):
    n = _cfg["prompts_per_theme"]
    return (
        f"Theme: {theme}\n"
        f"Please make all of these prompts about {theme} but make it so we do not fall on the same videos twice."
        f"I want videos with people entirely in the images so make the requests about actions mainly."
        f"Generate a numbered list of {n} AND ONLY {n} different prompts to use on Youtube in order to find videos based on {theme} in the following format.:\n"
        f"Answer in small letters and just give the actions in your ideas, no other context. Make it no more than 5 words."
        f"Do not forget: every answer must stick to {theme} in your responses."
        f"1:{theme}\n"
        f'2:"idea around theme 1" \n'
        f'3:"idea around theme 2" \n'
        f"..."
    )


if len(sys.argv) != 2:
    print("Usage: python YTpromptIdeas.py <video_theme>")
    sys.exit(1)

video_theme = sys.argv[1]
start_time = time.time()
prompt = get_video_theme_prompt(video_theme)
response = query_ollama_model(prompt)
end_time = time.time()
execution_time = end_time - start_time
print(prompt)
if response:
    print("Response:", response)

    titles = re.findall(r"^\s*\d+\:\s*(.*)$", response, flags=re.MULTILINE)
    print(titles)
    with open(_cfg["ideas_file"], "w") as file:
        for title in titles:
            file.write(f"{title}\n")  # Write a title in each line
    print("Search titles written in 'video_ideas.txt'.")
else:
    print("No response from the model.")
print(f"Execution time: {execution_time:.2f} seconds")
