import re
from conv_layer import ConvLayer

def parse_layers(layer_str):
    layers = layer_str.split('-')
    layer_arr = []
    for i, layer in enumerate(layers):
        if 'c' in layer:
            conv_channels = int(re.findall(r'\((.*?)\)', layer)[0])
            filter_size = int(layer[-2])
            cl = ConvLayer(filter_size, conv_channels, None)
            layer_arr.append(cl)
        elif 'p' in layer:
            pool_size = int(layer.replace('p', ''))
            layer_arr[-1].pool_size = pool_size
    return layer_arr