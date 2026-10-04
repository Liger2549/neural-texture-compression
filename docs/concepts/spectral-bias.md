# Spectral Bias

## First training
To store a texture in an MLP, the network takes the coordinate (u, v) of a texel as input and outputs its RGB colour. A sigmoid at the end clamps the output to [0, 1]. We train it on only this one texture and want it to overfit, because memorizing the texture is the compression: the weights become the texture. This model is the **plain MLP**.

## The problem
The result is blurry. The overall layout of the texture is visible, but all the fine detail is gone (22.31 dB).

![Plain MLP](../../results/Metal016/m2/plain/decoded.png)

The reason is **spectral bias**: an MLP learns smooth, low-frequency patterns first and sharp, high-frequency detail extremely slowly. Two neighbouring texels have almost the same input (e.g. u = 0.5000 and u = 0.5005), and a ReLU network's output changes smoothly with its input, so it gives them almost the same colour. A sharp edge, where the colour jumps between two neighbours, is very hard for it to learn. In theory a much bigger network trained much longer could get there, but in practice it stays blurry.

## Fix: Fourier features
Any image can be described as a sum of waves at different frequencies. So instead of giving the MLP (u, v) directly, we give it sin and cos of u and v at several frequencies: sin(2^k·π·u), cos(2^k·π·u) for k = 0 … 9.

- The low frequencies tell the network roughly *where* it is on the texture.
- The high frequencies change within a few texels, so neighbouring texels now get clearly different inputs, and the network can give them different colours.

With 10 frequencies, the highest one has a period of 4 texels, which matches the finest detail a 2048 texture can show. With the same network size, PSNR rises from 22.31 dB to 25.91 dB, and the patches and spots of the original appear.

![Plain vs Fourier vs reference](../../results/Metal016/m2/comparison.png)

## Limit of Fourier features
25.91 dB is still far from the BC7 baseline (47.82 dB), and close-up crops are still soft. The MLP has 208,643 weights and must memorize about 4 million texels of fine detail, which is too much for it. In M3 we store the detail in a **latent grid** (a small low-resolution texture of learned values), so the MLP only has to decode it instead of memorizing everything.
