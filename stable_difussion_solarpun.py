import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)  

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

num_cols = 3
num_rows = 4
prompt_1 = "digital illustration of human using a cylindrical food storage device in a solarpunk future, 8k resolution futuristic"
prompt_2 = "digital illustration of human using a cylindrical food storage device in a food cellar, 8k resolution futuristic solarpunk"
prompt_3 = "painting of large cylindrical rotating food storage device in in a solarpunk future in the style of studio ghibli, levels, sunset and green plants background"
prompt_4 = "painting of cylindrical rotating food storage device in a solarpunk kitchen in the style of studio ghibli, sunset and green plants background"

prompt = [prompt_1] * num_cols

all_images = []
for i in range(num_rows):
  images = pipe(prompt, num_inference_steps = 125).images
  all_images.extend(images)

grid = image_grid(all_images, rows=num_rows, cols=num_cols)
