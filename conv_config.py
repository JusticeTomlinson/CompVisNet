
one_layer_configs = [ 
    (1, 10, 3, 1, 1, 1),
    (1, 20, 3, 1, 1, 1),
    (1, 30, 3, 1, 1, 1),
    (1, 40, 3, 1, 1, 1),
    (1, 50, 3, 1, 1, 1),
    (1, 60, 3, 1, 1, 1),
    (1, 70, 3, 1, 1, 1),
    (1, 80, 3, 1, 1, 1),
    (1, 90, 3, 1, 1, 1),
    (1, 100, 3, 1, 1, 1)]

two_layer_configs = [
    (1, out_ch1, 3, 1, 1, 1, out_ch1, out_ch2, 3, 1, 1, 1) 
    for out_ch1 in range(10, 101, 10)  # First layer's out_channels
    for out_ch2 in range(20, 111, 10)  # Second layer's out_channels
]


three_layer_configs = [
    (1, out_ch1, 3, 1, 1, 1, out_ch1, out_ch2, 3, 1, 1, 1, out_ch2, out_ch3, 3, 1, 1, 1)
    for out_ch1 in range(10, 61, 10)  # First layer's out_channels
    for out_ch2 in range(20, 71, 10)  # Second layer's out_channels
    for out_ch3 in range(30, 81, 10)  # Third layer's out_channels
]
