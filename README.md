<div align = center>

![logo](assets/icon.gif)

# ClashState

VERSION 0.8

(c)2020-2024 By Pablo Roldán(Durandal)
</div>

---
# Table of contents

1. Introduction
2. Features

---

# 1 Introduction

*ClashState* is an image converter which allows to convert JPEG, GIF and PNG images to the native formats of 8-bit computers.

The software is written in Python and uses _DearPyGUI_ for the GUI.

The conversion algorithm started as a module I developed for [RetroBBS](https://github.com/retrocomputacion/retrobbs), it makes use of *HitherDither* de *Henrik Blidh* (with some changes), Pillow and Kmeans via OpenCV.

# 2 Features

### Supports the following graphic modes:

   - **Commodore 64 HiRes**: 320×200 pixels 16 colors – 2 colors per 8×8 pixels attribute.
   - **Commodore 64 Multicolor**: 160×200 pixels 16 colors – 4 colors per 4×8 pixels attribute, 1 of them fixed for the whole image.
   - **Commodore 64 AFLI**: 296×200 pixels 16 colors – 2 colors per 8×1 pixels attribute.
   - **Commodore 64 Unrestricted**: Fantasy mode, 320×200 pixels 16 colors, no restrictions.
   - **Commodore Plus/4 HiRes**: 320x200 pixels 121 colors - 2 colors per 8x8 pixels attribute.
   - **Commodore Plus/4 Multicolor**: 160x200 pixels 121 colors - 4 colors per 4x8 pixels attribute, 2 of them fixed for the whole image.
   - **Commodore Plus/4 Unrestricted**: Fantasy mode, 320x200 pixels 121 colors, no restrictions.
   - **MSX 1 Screen 2**: 256×192 pixels 15 colors – 2 colors per 8×1 pixels attribute.
   - **MSX 1 Unrestricted**: Fantasy mode, 256×192 pixels 15 colors, no restrictions.
   - **ZX Spectrum**: 256×192 pixels 15 colors – 2 colors per 8×8 pixels attribute.
   - **ZX Spectrum Unrestricted**: Fantasy mode, 256×192 pixels 15 colors, no restrictions.

![macaw](img/macaw_schouman.jpg)</br>Original - Macaw - Aert Schouman
| | |
|:---:|:---:
|![c64hires](img/macaw.png)</br>C64 Hires|![c64multi](img/macaw-multi.png)</br>C64 Multicolor
|![afli](img/macaw-afli.png)</br>C64 AFLI|![c64un](img/macaw_c64u.png)</br>C64 Unrestricted
|![p4hires](img/macaw_p4hi.png)</br>Plus/4 Hires|![p4multi](img/macaw_p4m.png)</br>Plus/4 Multicolor
|![p4u](img/macaw_p4u.png)</br>Plus/4 Unrestricted|![msxsc2](img/macaw-sc2.png)</br>MSX1 Unrestricted
|![msxu](img/macaw-msxu.png)</br>MSX1 Unrestricted|![zxsp](img/macaw_sp.png)</br>ZX Spectrum
|![zxspu](img/macaw_spu.png)</br>ZX Spectrum Unrestricted

Free positioning and scaling of the input image: Click and drag the input image to set the position/cropping. Use your mouse wheel for zooming.

### Real time adjustment of:

   - Contrast
   - Brightness
   - Hue
   - Saturation
   - Sharpness

### Dithering, color quantization:

   - 2-step quantization: luminance and final. The optional luminance step is applied only over the input image luminance channel and either the selected color palette’s luminance channel or over black and white.
   - Ten types of dithering/quantization:
     -   None: Nearest color, or no effect when selected for the luminance step
     -   Bayer 2×2
     -   Bayer 4×4
     -   Bayer 4×4 (Odd)
     -   Bayer 4×4 (Even)
     -   Bayer 4×4 (Spotty)
     -   Bayer 8×8
     -   Yliluoma
     -   Cluster dot
     -   Floyd Steinberg
    
   - With the exception of Yliluoma and Floyd Steinberg, the quantization threshold can be adjusted from 1 (darker) to 5 (lighter).

</br>

![Original](img/P61.jpg)</br>Original - P-61 y A-20 (Source: Smithsonian National Air and Space Museum)
| | |
|:---:|:---:
| ![none](img/P61-none.png)</br>C64 Hires None | ![bayer2x2](img/P61-2x2.png)</br>C64 Hires Bayer 2x2
| ![bayer4x4](img/P61-4x4.png)</br>C64 Hires Bayer 4x4 | ![bayer4x4even](img/P61-4x4even.png)</br>C64 Hires Bayer 4x4 Even
| ![bayer4x4odd](img/P61-4x4odd.png)</br>C64 Hires Bayer 4x4 Odd | ![bayer4x4spotty](img/P61-4x4spotty.png)</br>C64 Hires Bayer 4x4 Spotty
| ![bayer8x8](img/P61-8x8.png)</br>C64 Hires Bayer 8x8 | ![cluster](img/P61-cluster.png)</br>C64 Multicolor Cluster
| ![Yliluoma](img/P61-yliluoma.png)</br>C64 Multicolor Yliluoma | ![floyd-steinberg](img/P61-fs.png)</br>C64 Hires Floyd Steinberg

   - Selection for _Euclidean distance_, _CCIR 601_ or _LAb delta CIEDE 2000_ color comparison (found in the Options menu). Ylilouma and Floyd Steinberg use their own methods.

![Endeavour](/img/Endeavour_Space_Shuttle_Blastoff_(1055810551).jpg)</br>Original - Endeavour - Steve Jurvetson
| | |
|:---:|:---:
|![euclidean](/img/endeavour_euc.png)</br>MSX1 Euclidean distance |![CCIR 601](/img/endeavour_ccir.png)</br>MSX1 CCIR 601
|![CIEDE 2000](/img/endeavour_ciede.png)</br>MSX CIEDE 2000

### Color Palettes:

   - Quantization and preview palettes can be independently selected
    Each color in the palette can be enabled/disabled individually, for fine tuning the conversion result.
   - When enabling only 2 colors the quantization can be made over the selected color values or, use black and white for the quantization palette and the selected colors for the final image render.

![Original](/img/paul_cignac_saint-tropez.jpeg)
</br>Original - Paul Cignac - Saint Tropez
| | |
|:---:|:---:|
| ![full](/img/saint_full.png)</br>MSX full palette | ![yellow](/img/saint-yel.png)</br>MSX 5 active colors
| ![purple](/img/saint-purp.png)</br>MSX only purple and grey active | ![purplebw](/img/saint-pbyn.png)</br>MSX only purple and grey active, b&w mode

### Output formats:

   - Commodore 64 Hires:
     - Art Studio
     - C64 executable
     - PNG
   - Commodore 64 Multicolor:
     - Koala Paint
     - C64 executable
     - PNG
   - Commodore 64 AFLI:
     - AFLI editor
     - C64 executable
     - PNG
   - Commodore 64 unrestricted:
     - PNG
   - Commodore Plus/4 Hires:
     - Botticelli
     - Commodore Plus/4 executable
     - PNG
   - Commodore Plus/4 Multicolor:
     - Multi Botticelli
     - Commodore Plus/4 executable
     - PNG
   - Commodore Plus/4 unrestricted:
     - PNG
   - MSX1 Screen 2:
     - Screen 2
     - PNG
   - MSX1 unrestricted:
     - PNG
   - ZX Spectrum:
     - ZX Spectrum screen (.scr)
     - PNG
   - ZX Spectrum unrestricted:
     - PNG
     