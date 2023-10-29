import argparse
import json
import sys
import time

import diffusers
import torch

parser = argparse.ArgumentParser(
    description="Usage: python any3.py <name> --seed <seed> --steps <steps>"
)
parser.add_argument("name", type=str, help="生成するキャラクター名を指名")
parser.add_argument(
    "--seed", default=int(time.time()), type=int, help="default=unixtime"
)
parser.add_argument("--steps", default=20, type=int, help="default=20")
args = parser.parse_args()

with open("./prompts.json", "r") as file:
    data = json.load(file)

name = args.name
seed = args.seed
steps = args.steps
torch.manual_seed(seed)


def get_prompt_by_name(name):
    characters = data.get("characters", [])
    for character in characters:
        if character.get("name") == name:
            return character.get("prompt")
    raise ValueError(f"Character with name {name} not found.")


pipe = diffusers.StableDiffusionPipeline.from_pretrained("Linaqruf/anything-v3.0")
# pipe = diffusers.StableDiffusionPipeline.from_pretrained("xyn-ai/anything-v4.0")
# pipe.scheduler = diffusers.DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
pipe.load_textual_inversion(
    "./EasyNegative.safetensors",
    weight_name="EasyNegative.safetensors",
    token="EasyNegative",
)

prompt = get_prompt_by_name(name)
negative_prompt = "EasyNegative"
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_inference_steps=steps,
)
result.images[0].save("img_" + name + "_" + str(seed) + ".png")
