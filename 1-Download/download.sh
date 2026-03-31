#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate filterfusion

# You'll need to provide the search terms before in videos_ideas.txt.
# We have a script named YTPromptIdeas.py that can help you generate those ideas using an LLM.

# Duration and paths are read from config.yml at the repo root.
# You can still override them via CLI: python YTscrap.py <output_folder> <duration_in_minutes>

python YTscrap.py --verbose

python YTcut.py --keep

conda deactivate