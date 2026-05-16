from omegaconf import OmegaConf

from modeling.tokenizer import BAR_FSQ

config = OmegaConf.load("configs/generator/bar_b.yaml")
tokenizer = BAR_FSQ(config)
image_list = tokenizer.texts_to_image(["The quick brown fox jumps over the lazy dog."])
image = image_list[0]
image.save("output.png")
