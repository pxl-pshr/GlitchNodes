# ComfyUI GlitchNodes

GlitchNodes is a collection of image processing nodes designed for ComfyUI that specializes in creating glitch art and retro effects.

These are highly experimental and may break or even error. You are more than welcome to contribute to the repo.


                          :::!~!!!!!:.
                      .xUHWH!! !!?M88WHX:.
                    .X*#M@$!!  !X!M$$$$$$WWx:.
                   :!!!!!!?H! :!$!$$$$$$$$$$8X:
                  !!~  ~:~!! :~!$!#$$$$$$$$$$8X:
                 :!~::!H!<   ~.U$X!?R$$$$$$$$MM!
                 ~!~!!!!~~ .:XW$$$U!!?$$$$$$RMM!
                   !:~~~ .:!M"T#$$$$WX??#MRRMMM!
                   ~?WuxiW*`   `"#$$$$8!!!!??!!!
                 :X- M$$$$       `"T#$T~!8$WUXU~
                :%`  ~#$$$m:        ~!~ ?$$$$$$
              :!`.-   ~T$$$$8xx.  .xWW- ~""##*"
    .....   -~~:<` !    ~?T#$$@@W@*?$$      /`
    W$@@M!!! .!~~ !!     .:XUW$W!~ `"~:    :
    #"~~`.:x%`!!  !H:   !WM$$$$Ti.: .!WUn+!`
    :::~:!!`:X~ .: ?H.!u "$$$B$$$!W:U!T$$M~
    .~~   :X@!.-~   ?@WTWo("*$$$W$TH$! `
    Wi.~!X$?!-~    : ?$$$B$Wu("**$RM!
    $R@i.~~ !     :   ~$$$$$B$$en:``
    ?MXT@Wx.~    :     ~"##*$$$$M~
    ▒█▀▀█ ▀▄▒▄▀ ▒█░░░ ▒█▀▀█ ▒█▀▀▀█ ▒█░▒█ ▒█▀▀█
    ▒█▄▄█ ░▒█░░ ▒█░░░ ▒█▄▄█ ░▀▀▀▄▄ ▒█▀▀█ ▒█▄▄▀
    ▒█░░░ ▄▀▒▀▄ ▒█▄▄█ ▒█░░░ ▒█▄▄▄█ ▒█░▒█ ▒█░▒█
    ~~~~~~~~~ G L I T C H N O D E S ~~~~~~~~~

# Pixel8Bit Node

A comprehensive retro pixelation node that converts images to authentic 8-bit style with period-accurate palettes and dithering.
Supports both single images and batched images (N,H,W,C) with progress tracking.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/cJmyNT4s/Comfy-UI-34194.png" width="400">
</td>
<td>
<img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzkxM3B5NW9zN281MjBsM3VyZnduejhmazcwamt3NTU5ZjdmZXEyZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7qiofj8YgUkx4toHdb/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Core Settings
| Parameter | Description |
|-----------|-------------|
| pixel_size | Pixelation factor (1-256). Higher values create larger pixels. Default: 4 |
| quant_mode | Color quantization method: "Fixed Palette" or "Adaptive K-Means" |
| gamma | Gamma correction (0.1-3.0). Adjusts brightness curve before processing. Default: 1.0 |

### Palette Controls
| Parameter | Description |
|-----------|-------------|
| fixed_palette | Historical palette to use (see available palettes below) |
| custom_palette_hex | Custom colors in hex format (comma-separated). Used when palette is "Custom" |
| k_colors | Number of colors for K-Means mode (2-256). Default: 8 |

### Dithering
| Parameter | Description |
|-----------|-------------|
| dithering | Dithering algorithm: "None", "Ordered", or "Floyd-Steinberg" |
| ordered_size | Matrix size for ordered dithering: 2, 4, or 8. Default: 4 |
| posterize_bits | Bit depth reduction (1-8). Reduces color precision before quantization. Default: 8 |

## Available Palettes

<details>
<summary><strong>Classic Console Palettes</strong></summary>

- **PICO-8**: Fantasy console 16-color palette
- **GameBoy**: Original DMG-01 4-shade green palette
- **NES**: Nintendo Entertainment System color subset
- **C64**: Commodore 64 VIC-II 16 colors
- **Atari2600-Subset**: Common 16-color subset from Atari 2600
- **MSX**: TMS9918 palette from MSX computers

</details>

<details>
<summary><strong>Computer Palettes</strong></summary>

- **ZX Spectrum**: Sinclair ZX Spectrum 15 colors (7 base + bright variants)
- **Apple II**: Composite artifact colors
- **EGA-16**: IBM Enhanced Graphics Adapter 16 colors
- **VGA-256**: Standard 256-color VGA palette (6×6×6 RGB cube + grayscale)
- **Amiga-Workbench**: Popular Workbench/demoscene colors
- **CGA-Default**: IBM Color Graphics Adapter (cyan/magenta/white)
- **CGA-Alternate**: CGA alternate palette (green/red/yellow)

</details>

<details>
<summary><strong>Custom Options</strong></summary>

- **Custom**: Define your own palette using hex colors
  - Format: `#FF0000,#00FF00,#0000FF` (comma-separated)
  - Supports 3-digit and 6-digit hex codes

</details>

## Quantization Modes

### Adaptive K-Means
Analyzes the image to create an optimal palette:
- **Content-Aware**: Colors chosen based on image content
- **Customizable**: Control exact number of colors (2-256)
- **Flexible**: Works with any image while maintaining retro feel

## Dithering Options

### None
Direct color quantization without dithering:
- **Clean**: Sharp color transitions
- **Minimal**: No additional patterns
- **Fast**: Quickest processing

### Ordered (Bayer Matrix)
Uses mathematical dither patterns:
- **2×2**: Basic checkerboard patterns
- **4×4**: Balanced detail and smoothness  
- **8×8**: Fine detail with smooth gradients
- **Consistent**: Repeatable patterns

### Floyd-Steinberg
Error diffusion dithering:
- **Organic**: Natural-looking transitions
- **Smooth**: Best gradient representation
- **Authentic**: Classic algorithm from early computing

##

# Corruptor Node

A node that applies controlled corruption effects to images using wavelet transformations.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/PrKFgyHc/Corruptor-Image-00001.png" width="400">
</td>
<td>
<img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3N4ZHRsczBldW9jY2N3OTNob2Rnd240M2FtZ2N3Mmhjcjd2dGJweSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/hipjAbeMgErWPXIUUA/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Core Settings
| Parameter | Description |
|-----------|-------------|
| scaling_factor_in | Controls initial corruption intensity (0-1000). Higher values create more subtle effects, lower values create more extreme effects. Default: 80.0 |
| scaling_factor_out | Controls final corruption intensity (0-1000). Higher values amplify the effect, lower values reduce it. Default: 80.0 |
| noise_strength | Amount of random noise added (0-100). Higher values create more random distortions. Default: 10.0 |
| color_space | Color space for applying effects: RGB, HSV, LAB, or YUV. Each creates different corruption patterns |
| channels_combined | When True, processes all color channels together. When False, processes each channel independently |

### Advanced Wavelet Controls
| Parameter | Description |
|-----------|-------------|
| wavelet_floor_mode | Method used for coefficient quantization: "regular", "absolute", or "threshold". Default: "regular" |
| wavelet_padding | Padding mode for signal extension: "edge", "constant", "reflect", or "symmetric". Default: "edge" |
| wavelet_threshold | Threshold value used when wavelet_floor_mode is "threshold". Default: 0.5 |

### Optional Controls
| Parameter | Description |
|-----------|-------------|
| noise_distribution | Type of noise to add: "normal" (Gaussian), "uniform" (even), or "salt_pepper" (random extreme values). Default: "normal" |

##

# DataBend Node

Advanced glitch art node that combines multiple effects including slice manipulation, color shifting, and various distortion patterns.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/T3C728qW/Data-Bend-Image-00002.png" width="400">
</td>
<td>
<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzF4YTE5YWZtbjNxeTB2YTRiazE0cHNic3ZtZW45c3l3MzFhdjN6dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/BkvVWlB3BNmHZTY6wx/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Slice/Block Controls
| Parameter | Description |
|-----------|-------------|
| slice_direction | Direction of glitch slices: "horizontal", "vertical", or "both". Default: horizontal |
| slice_min_size | Minimum size of glitch slices (1-50). Default: 5 |
| slice_max_size | Maximum size of glitch slices (5-200). Default: 40 |
| slice_variability | Controls frequency of slice effects (0-1). Higher values create more slices. Default: 0.5 |

### Color Manipulation
| Parameter | Description |
|-----------|-------------|
| channel_shift_mode | Type of color shifting: "random", "rgb_split", or "hue_shift". Default: random |
| color_intensity | Strength of color manipulation effects (0-1). Default: 0.7 |
| rgb_shift_separate | When true, applies RGB shifting to channels independently. Default: False |
| preserve_bright_areas | Protects bright areas from color manipulation (0-1). Default: 0.5 |

### Glitch Pattern Controls
| Parameter | Description |
|-----------|-------------|
| glitch_types | Type of glitch effect: "shift", "repeat", "mirror", "noise", or "all". Default: all |
| pattern_frequency | Number of glitch patterns to apply (1-10). Default: 3 |
| chaos_amount | Intensity of glitch patterns (0-1). Default: 0.5 |
| seed | Random seed for reproducible results. -1 for random. Default: -1 |

### Distortion Controls
| Parameter | Description |
|-----------|-------------|
| wave_distortion | Applies wave-like distortion effect (0-1). Default: 0.0 |
| compression_artifacts | Simulates compression artifacts (0-1). Default: 0.0 |
| pixel_sorting | Applies pixel sorting effect (0-1). Default: 0.0 |

### Additional Controls
| Parameter | Description |
|-----------|-------------|
| control_after_generate | Post-processing control: "randomize" or "none". Default: none |

##

# DitherMe Node

A comprehensive dithering node that implements various dithering algorithms for creating retro and artistic effects.
Supports both single images and batched images (N,H,W,C) with temporal coherence for animations.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/pLgGxWG3/Comfy-UI-00008.png" width="400">
</td>
<td>
<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZWFobmVidWE5cWJieTM3MXh1NnhtZjh2YWtoNDNtcjFhNHN2cGlqayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/VxSuRNAporxNdRisUs/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Core Settings
| Parameter | Description |
|-----------|-------------|
| algorithm | Dithering algorithm to use. See full list below. Default: floyd_steinberg |
| color_mode | Output color scheme: "monochrome", "duotone", "tritone", or "indexed" |
| effect_size | Scale factor for dithering patterns (0.1-10.0). Affects pattern density and size. Default: 1.0 |
| threshold | Base threshold for dithering decisions (0-1). Default: 0.5 |

### Image Processing
| Parameter | Description |
|-----------|-------------|
| sharpen | Sharpening intensity (0-2.0). Applied before dithering. Default: 0.0 |
| blur | Blur intensity (0-2.0). Applied before dithering. Default: 0.0 |
| noise | Random noise amount (0-1.0). Adds texture before dithering. Default: 0.0 |
| temporal_coherence | Frame blending for animations (0-1.0). Reduces flickering in sequences. Default: 0.0 |

### Color Controls (Optional)
| Parameter | Description |
|-----------|-------------|
| shadow_color | Color for dark areas (hex format). Default: #000000 |
| midtone_color | Color for middle tones (hex format). Default: #808080 |
| highlight_color | Color for bright areas (hex format). Default: #FFFFFF |
| shadow_brightness | Brightness adjustment for shadows (-1 to 1). Default: 0.0 |
| midtone_brightness | Brightness adjustment for midtones (-1 to 1). Default: 0.5 |
| highlight_brightness | Brightness adjustment for highlights (-1 to 1). Default: 1.0 |
| palette_colors | Number of colors for indexed mode (2-256). Default: 2 |

## Available Algorithms

<details>
<summary><strong>Error Diffusion</strong></summary>

- `floyd_steinberg` - Classic Floyd-Steinberg dithering
- `jarvis_judice_ninke` - JJN algorithm with wider error distribution
- `stucki` - Stucki algorithm for smooth gradients
- `atkinson` - Atkinson dithering (used in early Mac computers)
- `burkes` - Burkes filter for balanced error distribution
- `sierra` - Sierra filter family (3 variants)
- `sierra_2row` - Two-row Sierra variant
- `sierra_lite` - Lightweight Sierra variant

</details>

<details>
<summary><strong>Ordered Dithering</strong></summary>

- `ordered_2x2`, `ordered_4x4`, `ordered_8x8` - Standard ordered matrices
- `bayer_2x2`, `bayer_4x4`, `bayer_8x8` - Bayer pattern matrices

</details>

<details>
<summary><strong>Clustered/Dispersed Patterns</strong></summary>

- `clustered_dot_4x4`, `clustered_dot_8x8` - Clustered dot patterns
- `dispersed_dot_4x4`, `dispersed_dot_8x8` - Dispersed dot patterns

</details>

<details>
<summary><strong>Special Effects</strong></summary>

- `halftone_dots` - Circular halftone patterns
- `halftone_lines` - Linear halftone patterns
- `modulation` - Smooth wave-based line modulation
- `wave_interference` - Flowing interference patterns
- `contour_lines` - Lines following image contours
- `line_modulation` - Clean modulated line patterns

</details>

<details>
<summary><strong>Noise & Spatial</strong></summary>

- `blue_noise`, `white_noise` - Noise-based dithering
- `random` - Random threshold dithering
- `threshold` - Simple threshold conversion

</details>

<details>
<summary><strong>Geometric Patterns</strong></summary>

- `hilbert_curve` - Space-filling curve patterns
- `spiral` - Spiral-based dithering
- `zigzag` - Zigzag scan patterns
- `checkerboard` - Checkerboard-based dithering

</details>

<details>
<summary><strong>Diffusion Variants</strong></summary>

- `diffusion_horizontal` - Horizontal-only error diffusion
- `diffusion_vertical` - Vertical-only error diffusion
- `diffusion_diagonal` - Diagonal error diffusion
- `void_and_cluster` - Advanced void-and-cluster algorithm

</details>

##

# OrderedDithering Node

A specialized ordered dithering node that creates precise, controllable dithering patterns with animation support.
Supports both single images and batched images (N,H,W,C) with temporal effects.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/vZhPgJzZ/Comfy-UI-00007.png" width="400">
</td>
<td>
<img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExeTVlbTRjdWwyaHFqZmtqdnF6bnJ2NjdmZzQ1dHV5ZnlhdXVrdmVxaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2QFww1dnMkY4ji5B0l/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Core Settings
| Parameter | Description |
|-----------|-------------|
| dither_type | Type of dithering: "Standard", "Artistic", or "Animated" |
| color_mode | Output format: "Color" (RGB) or "Grayscale" |
| num_colors | Number of quantization levels (2-256). Default: 2 |
| pattern_size | Dither matrix size: "2x2", "4x4", or "8x8" |
| scale | Pattern scaling factor (1-8). Enlarges the base pattern. Default: 1 |

### Visual Controls
| Parameter | Description |
|-----------|-------------|
| pattern_contrast | Pattern intensity (0.1-2.0). Higher values create stronger patterns. Default: 1.0 |
| invert | Inverts colors and patterns when enabled. Default: False |

### Animation Controls (Animated Mode)
| Parameter | Description |
|-----------|-------------|
| frames | Number of animation frames to generate (1-300). Default: 60 |
| speed | Base animation speed (0-1.0). Controls pattern shifting. Default: 0.0 |
| wave_speed | Wave modulation speed (0.1-2.0). Controls pattern flow. Default: 1.0 |

## Dithering Types

<details>
<summary><strong>Standard</strong></summary>

Uses traditional Bayer matrices for precise, mathematical dithering patterns:
- **2x2**: Basic checkerboard pattern, sharp transitions
- **4x4**: Balanced detail and smoothness
- **8x8**: Fine detail, smooth gradients

Best for: Technical applications, retro computer graphics, precise color reduction

</details>

<details>
<summary><strong>Artistic</strong></summary>

Enhanced patterns with creative variations:
- Dot patterns for texture
- Line patterns (horizontal/vertical)
- Diagonal patterns for dynamic effects
- Density variations based on brightness levels

Best for: Creative effects, stylized artwork, unique visual textures

</details>

<details>
<summary><strong>Animated</strong></summary>

Creates temporal dithering effects with moving patterns:
- Pattern offset animation
- Wave-based modulation
- Smooth transitions between frames
- Controllable speed and flow

Best for: Video effects, dynamic backgrounds, animated textures

</details>

## Pattern Matrices

The node uses optimized Bayer matrices:

**2x2 Matrix:**
```
0  2
3  1
```

**4x4 Matrix:**
```
0   8   2  10
12  4  14   6
3  11   1   9
15  7  13   5
```

**8x8 Matrix:** Full 64-element optimized Bayer pattern for maximum detail.

##

# GlitchIT Node

A node that creates JPEG corruption artifacts by manipulating image data at the byte level.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/1tJJk5hF/Glitch-It-Image-00006.png" width="400">
</td>
<td>
<img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNTgzcjM5OGRwOXVmc3podHVqbDY5ODQ5Ynlib3dnb3B4YWpvdGl1MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1Ps2nRakhIDKpbq1FX/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

| Parameter | Description |
|-----------|-------------|
| seed | Random seed for reproducible results. Controls the random selection of bytes to corrupt. Default: 0 |
| min_amount | Minimum number of bytes to corrupt (0-100). Default: 1 |
| max_amount | Maximum number of bytes to corrupt (1-100). Default: 10 |

##

# Interference Node

Advanced shader-like interference effect node that creates complex patterns through iterative sorting and color manipulation.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/7YGbJs68/Inter-Image-00011.png" width="400">
</td>
<td>
<img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZHdldzlzODkzbXV4cWwzNm42N295NzM3Y3dvdG13NnV5dzgwc3hiNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/OcQItoCjsQG9vo4fdO/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Pattern Controls
| Parameter | Description |
|-----------|-------------|
| horizontal_iterations | Number of horizontal sorting passes (0-50). Higher values create more horizontal interference. Default: 10 |
| vertical_iterations | Number of vertical sorting passes (0-50). Higher values create more vertical interference. Default: 4 |
| shift_amount | Pixel shift distance per iteration (-10 to 10). Negative values shift left/up, positive values shift right/down. Default: -1 |

### Color Controls
| Parameter | Description |
|-----------|-------------|
| color_shift | Intensity of color transformation (0-1). Higher values create more pronounced color effects. Default: 0.5 |
| color_mode | Color effect type: "monochrome", "rainbow", "duotone", or "invert". Each mode creates distinct color patterns |
| preserve_brightness | When enabled, maintains original image brightness levels while applying color effects. Default: True |

##

# LineScreen Node

An image processing node that creates halftone-like effects using configurable line patterns.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/9Q1fvvg9/Line-Screen-Image-00002.png" width="400">
</td>
<td>
<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXZmYnpxaDcxMzF6ZzFrdzRyMnlrNWxndnUwNzJ2b216aWptZ3g1bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/AfEFaveCKhC3MZHE3a/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Pattern Controls
| Parameter | Description |
|-----------|-------------|
| line_spacing | Distance between lines (2-20). Lower values create denser patterns. Default: 4 |
| angle | Rotation angle of the line pattern (-90° to 90°). Default: -45.0 |
| threshold | Brightness threshold for line application (0.1-0.9). Controls pattern density. Default: 0.5 |
| contrast_boost | Enhances image contrast before applying pattern (1.0-2.0). Default: 1.2 |
| invert | When enabled, inverts the pattern colors. Default: False |

### Line Color (RGB)
| Parameter | Description |
|-----------|-------------|
| line_color_r | Red component of line color (0-1). Default: 0.0 |
| line_color_g | Green component of line color (0-1). Default: 0.0 |
| line_color_b | Blue component of line color (0-1). Default: 0.0 |

### Background Color (RGB)
| Parameter | Description |
|-----------|-------------|
| bg_color_r | Red component of background color (0-1). Default: 1.0 |
| bg_color_g | Green component of background color (0-1). Default: 1.0 |
| bg_color_b | Blue component of background color (0-1). Default: 1.0 |

##

# LuminousFlow Node

A node that transforms images into flowing luminous strands, creating an ethereal effect of light threads that follow the image's features.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/tTzZ4Y2r/Lum-Image-00007.png" width="400">
</td>
<td>
<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExa244MzFlZzk5dThlNHo1YW1yOG9weG5oYTBrZWs0Z3FhYjdlY3MweiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/cxWDSSSRy1zT3zVUZD/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Line Controls
| Parameter | Description |
|-----------|-------------|
| line_spacing | Distance between luminous lines (2-50). Lower values create denser patterns. Default: 8 |
| line_thickness | Width of each luminous line (1-5). Default: 1 |
| flow_intensity | Strength of the flow effect (0.1-5.0). Higher values create more dramatic displacements. Default: 2.0 |
| smoothing | Amount of line smoothing (0-5). Higher values create smoother curves. Default: 1.0 |

### Glow Controls
| Parameter | Description |
|-----------|-------------|
| glow_intensity | Brightness of the luminous effect (0.1-20.0). Default: 12.0 |
| glow_spread | Size of the glow effect (0-12). Higher values create wider glows. Default: 7 |
| darkness | Background darkness level (0-1). Lower values create darker backgrounds. Default: 0.01 |

### Color Controls
| Parameter | Description |
|-----------|-------------|
| vibrancy | Color saturation intensity (0.1-15.0). Higher values create more vivid colors. Default: 8.0 |
| contrast | Overall contrast enhancement (0.1-10.0). Higher values create stronger distinction between light and dark areas. Default: 4.0 |

##

# PixelFloat Node

An advanced animation processor that creates fluid, gravity-affected motion between frames using optical flow analysis.

## Example

<table>
<tr>
<td>
<img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGh5dXd2cHNhczdraXg1dDJnZzRybjNlb29wcXJ5bmYybTZ1MWZnaSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/KKQtKITq4q2kpGjhfQ/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts image batch: ✓ (Processes transitions between consecutive frames)

## Parameters

### Motion Controls
| Parameter | Description |
|-----------|-------------|
| gravity_strength | Strength of downward motion (-50 to 0). More negative values create stronger gravity effect. Default: -10.0 |
| motion_threshold | Threshold for motion detection (0.1-5.0). Higher values require more motion to trigger effects. Default: 0.5 |
| interpolation_factor | Smoothness of motion transitions (0-1). Higher values create smoother transitions. Default: 0.5 |

### Block Configuration
| Parameter | Description |
|-----------|-------------|
| block_size | Size of pixel blocks for motion analysis (4-64). Default: 4 |
| auto_block_size | When enabled, automatically calculates optimal block size. Default: False |
| min_blocks | Minimum number of blocks when using auto sizing (16-64). Default: 32 |
| max_blocks | Maximum number of blocks when using auto sizing (64-256). Default: 128 |

### Flow Analysis
| Parameter | Description |
|-----------|-------------|
| flow_scale | Scale factor for optical flow pyramid (0.1-0.9). Default: 0.25 |
| flow_levels | Number of pyramid levels for flow analysis (1-8). Default: 5 |
| flow_iterations | Number of iterations per pyramid level (1-10). Default: 3 |

##

# PixelRedistribution Node

A node that creates unique visual effects by redistributing pixels based on various patterns and color relationships.

## Example

<table>
<tr>
<td>
<img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3Uxem83aDJvM3dqdWduZ2M0cHJqdXZ5bDZncmZqaGZ2cDF5emhoayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bufgvJ1Lmtd12jcsmP/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts image batch: ✓

## Parameters

### Pattern Controls
| Parameter | Description |
|-----------|-------------|
| distance_mode | Origin point for redistribution: "center", "top", "left", or "random". Default: center |
| pattern | Distribution pattern: "outward", "spiral", "waves", or "diagonal". Default: outward |
| strength | Intensity of the redistribution effect (0.1-2.0). Default: 1.0 |

### Color Controls
| Parameter | Description |
|-----------|-------------|
| color_size | Number of color levels per channel (2-256). Higher values create smoother gradients. Default: 64 |
| order | Channel processing order as comma-separated RGB indices (e.g., "0,1,2"). Default: "0,1,2" |
| invert | When enabled, inverts the color output. Default: False |

### Adjustments
| Parameter | Description |
|-----------|-------------|
| contrast | Adjusts image contrast before processing (0.1-4.0). Default: 1.0 |
| brightness | Adjusts image brightness before processing (-1.0 to 1.0). Default: 0.0 |

##

# Rekked Node

A versatile image processing node that applies various glitch and artistic effects through different processing modes.
Original Repo: https://github.com/Datamosh-js/datamosh

## Example

<table>
<tr>
<td>
<img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3k3eGQ4d3N0cXRkYndnbzg1MzFkcTlzMTh2OHlvenlocnhxY2t0bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ei83Y3lU7fTntgHvoX/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

| Parameter | Description |
|-----------|-------------|
| mode | Choose from the following effect modes: |

### Available Modes
- **blurbobb**: Creates random pixel blocks at intervals
- **fatcat**: Intensifies colors with layered brightness multiplication
- **vaporwave**: Applies vaporwave aesthetic with specific color palette
- **castles**: Creates high contrast masks based on luminance thresholds
- **chimera**: Combines channel mixing with noise and grain effects
- **gazette**: Applies selective pixel blocking with binary threshold
- **manticore95**: Creates complex distortion patterns with pixel shifting
- **schifty**: Generates random-length block shifting effects
- **vana**: Applies selective color channel mixing with randomization
- **veneneux**: Creates poisonous-looking color shifts with seed variation
- **void**: Combines noise and grain for dark, textured effects
- **walter**: Applies threshold-based color transformations with balance correction

##

# Scanz Node

A comprehensive glitch effect node that combines scan lines, wave distortions, and color manipulation to create CRT and digital artifact effects.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/43WR4Gp6/Scanz-Image-00007.png" width="400">
</td>
<td>
<img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMnprN2pwNmkxNXJheDgwZjN6eXhqZ2YxZ3BzZWcwZWg0NGlhaXJ5YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kyTn8U2T0WE0zjHElo/giphy.gif" width="400">
</td>
</tr>
</table

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Base Effects
| Parameter | Description |
|-----------|-------------|
| glitch_amount | Controls overall compression artifacts (0-1). Higher values create more JPEG-like distortion. Default: 0.5 |
| channel_shift | Amount of RGB channel displacement (0-1). Creates chromatic aberration effects. Default: 0.2 |
| pixel_sorting | Threshold for brightness-based pixel sorting (0-1). Default: 0.3 |

### Wave Distortions
| Parameter | Description |
|-----------|-------------|
| wave_amplitude | Intensity of wave distortion (0-1). Default: 0.0 |
| wave_frequency | Frequency of wave patterns (0.1-5.0). Higher values create tighter waves. Default: 0.5 |
| wave_speed | Speed of wave movement (0.1-5.0). Affects distortion pattern spacing. Default: 1.0 |

### Scan Lines
| Parameter | Description |
|-----------|-------------|
| scan_lines | Intensity of CRT-like scan lines (0-1). Default: 0.0 |
| scan_drift | Amount of horizontal drift in scan lines (0-1). Default: 0.0 |
| scan_curve | CRT-style screen curvature effect (0-1). Default: 0.2 |

### Color Effects
| Parameter | Description |
|-----------|-------------|
| color_drift | Vertical color shifting intensity (0-1). Default: 0.0 |
| static_noise | Amount of static noise overlay (0-1). Darker areas receive more noise. Default: 0.0 |
| edge_stretch | Edge-based pixel stretching intensity (0-1). Default: 0.0 |

##

# TvGlitch Node (WIP Extremely Long Processing)

Simulates authentic analog TV signal distortions and artifacts by emulating the YIQ color space and composite video signal characteristics.

## Example

<table>
<tr>
<td>
<img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGF5OHVkNXhzcHMybXRieW45YTRiZWE0NDVoZjhpYXd0YXlmdmZpcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GkdjywvoPfMJ5BwuFO/giphy.gif" width="400">
</td>
</tr>
</table>


## Input Type
- Accepts image batch: ✓

## Parameters

### Signal Parameters
| Parameter | Description |
|-----------|-------------|
| subcarrier_amplitude | Strength of the color subcarrier signal (1-200). Higher values increase color bleeding. Default: 40 |
| composite_preemphasis | Enhances high-frequency components (0-100). Simulates TV signal pre-emphasis. Default: 1.0 |

### Noise Effects
| Parameter | Description |
|-----------|-------------|
| video_noise | Amount of luminance noise (0-10000). Affects overall brightness fluctuations. Default: 100 |
| video_chroma_noise | Amount of color noise (0-10000). Creates color distortions. Default: 100 |
| video_chroma_phase_noise | Color phase shifting intensity (0-100). Causes color wobbling. Default: 15 |

### Display Effects
| Parameter | Description |
|-----------|-------------|
| video_chroma_loss | Probability of color signal dropout per scanline (0-1). Creates color loss effects. Default: 0.24 |
| scanlines_scale | Intensity of CRT scanline effect (0-5). Higher values create more pronounced scanlines. Default: 1.5 |

##

# VaporWave Node

A stylistic effect node that transforms images into vaporwave aesthetics using customizable color bands and thresholds.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/XqcTh0NB/Vwave-Image-00009.png" width="400">
</td>
<td>
<img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExa2dpYmJsbXhzcGJuZW1xeWxyNXRremhyeXhrczY1d2Yzd3pvMzFheiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kBi44iwyIa2dEsbyr8/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Threshold Controls
| Parameter | Description |
|-----------|-------------|
| threshold_dark | Darkest threshold level (0-1). Pixels below this become black. Default: 0.059 |
| threshold_light | Lightest threshold level (0-1). Pixels above this become white. Default: 0.922 |
| mid_threshold_1 | First mid-level threshold (0-1). Default: 0.235 |
| mid_threshold_2 | Second mid-level threshold (0-1). Default: 0.471 |
| mid_threshold_3 | Third mid-level threshold (0-1). Default: 0.706 |

### Color Band Controls

#### Color 1 (Cyan Band)
| Parameter | Description |
|-----------|-------------|
| color1_r | Red component (0-1). Default: 0.0 |
| color1_g | Green component (0-1). Default: 0.722 |
| color1_b | Blue component (0-1). Default: 1.0 |

#### Color 2 (Magenta Band)
| Parameter | Description |
|-----------|-------------|
| color2_r | Red component (0-1). Default: 1.0 |
| color2_g | Green component (0-1). Default: 0.0 |
| color2_b | Blue component (0-1). Default: 0.757 |

#### Color 3 (Purple Band)
| Parameter | Description |
|-----------|-------------|
| color3_r | Red component (0-1). Default: 0.588 |
| color3_g | Green component (0-1). Default: 0.0 |
| color3_b | Blue component (0-1). Default: 1.0 |

#### Color 4 (Aqua Band)
| Parameter | Description |
|-----------|-------------|
| color4_r | Red component (0-1). Default: 0.0 |
| color4_g | Green component (0-1). Default: 1.0 |
| color4_b | Blue component (0-1). Default: 0.976 |

##

# VHSonAcid Node

A psychedelic image processing node that combines VHS-style artifacts with acid-like color distortions.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/pLTTtPhH/VHSAcid-Image-00009.png" width="400">
</td>
<td>
<img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExc25ncjVyYm85NnNtc3lya29mMnB6N2VycnA0Mzc0eW9yOWtoaW5veCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xnQAAjASECqnrgZQ9h/giphy.gif" width="400">
</td>
</tr>
</table>

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Glitch Controls
| Parameter | Description |
|-----------|-------------|
| slice_size | Height of glitch slices (1-100). Larger values create thicker bands of distortion. Default: 20 |
| offset_range | Maximum horizontal displacement of slices (1-200). Larger values allow for more extreme shifts. Default: 50 |
| glitch_probability | Chance of applying glitch to each slice (0-1). Higher values create more frequent distortions. Default: 0.3 |

### Color Effect
| Parameter | Description |
|-----------|-------------|
| color_shift | Intensity of RGB channel separation (0-1). Higher values create more extreme color bleeding. Default: 0.5 |

##

# VideoModulation Node

A node that simulates CRT monitor effects with scan lines, RGB shift, and dot patterns.

## Examples

<table>
<tr>
<td>
<img src="https://i.postimg.cc/85hkb53h/Vid-Mod-Image-00010.png" width="400">
</td>
<td>
<img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ2d4a2NwbnRsZjFzbm41NnR5NmU0amJlcm93aGh1NXBiNmp0NHRqYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Ly5fHIcuGJbXvIJhs4/giphy.gif" width="400">
</td>
</tr>
</table

## Input Type
- Accepts single image: ✓
- Accepts image batch: ✓

## Parameters

### Display Effects
| Parameter | Description |
|-----------|-------------|
| scan_density | Frequency of scan lines (2-10). Higher values create more tightly packed lines. Default: 4 |
| dot_pattern | Intensity of CRT-style dot matrix effect (0-1). Higher values create more visible dots. Default: 0.15 |

### Color Adjustments
| Parameter | Description |
|-----------|-------------|
| rgb_shift | Amount of RGB channel separation (0-0.05). Creates chromatic aberration effect. Default: 0.015 |
| brightness | Overall brightness multiplier (0.5-2.0). Enhances the CRT glow effect. Default: 1.2 |

