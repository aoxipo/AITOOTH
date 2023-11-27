from util import get_level_set
from CLAHE import equalize_clahe
import torch
import torch.nn as nn

def gamma(img, alpha = 2.3):
    invalpha = 1/alpha
    img = torch.pow(img , invalpha) 
    return img

class DataWarper(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clahe = equalize_clahe
        self.level_set_method = get_level_set

    @torch.no_grad()
    def forward(self, x, y):
        x = self.clahe(x)
        level_set = self.level_set_method(y)
        return x, level_set