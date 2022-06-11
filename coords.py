from pydantic import BaseModel
from typing import List

class coord(BaseModel):
    x: float 
    y: float 


class coord_list(BaseModel):
    num_examples: int 
    inputs: List[coord] 
    