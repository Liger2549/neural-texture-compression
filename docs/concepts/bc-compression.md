# BC (Block Compression)

## What it is
BC is a lossy texture compression format built into every GPU. It lets textures use
less memory while the GPU can still read any pixel instantly.

## The problem
- Memory: an uncompressed 2K roughness map is 4.19 MB. A game has thousands of textures.
- Random access: formats like PNG can't be read pixel by pixel. BC can, because every tile has the same size (8 bytes for BC4), so the GPU can calculate where any pixel's tile is and read only that tile, like indexing an array.

## How one block works (BC4 example)
1. The image is cut into 4×4 tiles.
2. Each tile stores 2 values: minimum value and maximum value.
3. The GPU makes 8 evenly spaced values between them (the palette).
4. Each pixel stores only the index of the closest palette value (3 bits).
Result: 16 pixels in 8 bytes = 4 bits per pixel instead of 8.

## What I measured (Metal016, M1)
| Map | Format | bpt | PSNR |
|---|---|---|---|
| albedo | BC7 | 8 | 47.82 |
| normal | BC5 | 8 | 47.01 |
| roughness | BC4 | 4 | 52.18 |
| metalness | BC4 | 4 | 41.08 |
| Total | | 24.0 | |

Metalness has the lowest PSNR because tiles on the edge of a non-metal patch contain both 0 and 255. The palette must cover the whole range, so its 8 steps are about 36 apart, and in-between values snap to a wrong step.

## Why NTC can do better
In Metal016, the reddish patches change albedo, roughness and metalness at the same place.
BC compresses each map separately, so it stores that patch shape 3 times.
NTC decodes all channels together from shared features, so it can store it once.
The goal: similar quality (40+ dB) at fewer than 24 bits per texel.
