from .Fast_ACV import Fast_ACVNet
from .Fast_ACV_plus import Fast_ACVNet_plus
from .Fast_ACV_plus2 import Fast_ACVNet_plus2
from .Fast_ACV_plus_refine import Fast_ACVNet_plus_refine

__models__ = {
    'Fast_ACVNet': Fast_ACVNet,
    'Fast_ACVNet_plus' : Fast_ACVNet_plus,
    'Fast_ACVNet_plus2' : Fast_ACVNet_plus2,
    'Fast_ACVNet_plus_refine' : Fast_ACVNet_plus_refine
}