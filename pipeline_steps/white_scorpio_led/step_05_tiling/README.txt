Step 5 - Tiling and flip augmentation
=====================================

Each scaled image is cut into 768 x 768 px tiles with 192 px overlap; every tile also runs mirrored (horizontal flip) and the two predictions are averaged. Tiles are blended with a Hann window so there are no seams.
grid_*.png shows the tile positions; sample_tile*.png is one tile and its mirrored copy.
