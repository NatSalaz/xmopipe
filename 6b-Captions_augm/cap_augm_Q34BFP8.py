import os
import json
import yaml
import numpy as np
import random
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yml"))["captions_augm"]

model_name = _cfg["model_name"]

print("\nChargement du modèle Qwen3...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

root_dir = _cfg["npz_root"]
video_dirs = [
    os.path.join(root_dir, d)
    for d in os.listdir(root_dir)
    if d.startswith("video_") and os.path.isdir(os.path.join(root_dir, d))
]

for i in range(_cfg["num_samples"]):
    data_npz = {}
    while len(list(data_npz.keys())) != 2:
        video_dir = random.choice(video_dirs)
        video_id = os.path.basename(video_dir)
        npz_files = [f for f in os.listdir(video_dir) if f.endswith(".npz")]
        if not npz_files:
            raise RuntimeError(f"Aucun fichier NPZ trouvé dans {video_dir}")
        chosen_npz = random.choice(npz_files)
        npz_path = os.path.join(video_dir, chosen_npz)
        data_npz = np.load(npz_path, allow_pickle=True)

    print("Fichier NPZ choisi :", npz_path)
    print("Clés du NPZ :", list(data_npz.keys()))

    # Load text from JSON
    json_path = os.path.join(root_dir, video_dir, f"description_videos_{video_id}.json")

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        descriptions = json.load(f)

    scene_number = None
    part = chosen_npz.replace(".npz", "").split("_")
    scene_number = part[-1]

    if scene_number is None:
        raise ValueError("NO SCENE NUMBER")

    json_key = f"{video_id}/{scene_number}"
    print("\nClé utilisée pour le JSON :", json_key)

    data_txt = descriptions.get(json_key, "Aucune description trouvée.")
    print("\nTexte associé :\n", data_txt)

    prompt = f"Section numbers 0 and 1 describe 2 different people. Make 1 sentences with a few words describing what the people are doing, the environment and the body description and style of movement: {data_txt}. Do it 4 times with different words, do not be TOO specific about environment. Follow this format: 1. The first person is [...] while the other one is [...]  in environment description, etc."
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nGénération en cours...")
    generated_ids = model.generate(**model_inputs, max_new_tokens=_cfg["max_new_tokens"])
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    lines = content.split("\n")
    clean_lines = []
    for line in lines:
        line = line.lstrip("0123456789. ").strip()
        if line:
            clean_lines.append(line)

    # One sentence per line, no blank lines
    final_text = "\n".join(clean_lines)

    # Affichage
    if thinking_content != "":
        print("\n" + "=" * 60)
        print("THINKING CONTENT:")
        print("=" * 60)
        print(thinking_content)

    print("\n" + "=" * 60)
    print("FINAL CONTENT:")
    print("=" * 60)
    print(final_text)
    with open(f"{_cfg['output_dir']}/{video_id}_{scene_number}.txt", "w") as text_file:
        text_file.write(final_text)
