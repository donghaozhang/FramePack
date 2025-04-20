<p align="center">
    <img src="https://github.com/user-attachments/assets/2cc030b4-87e1-40a0-b5bf-1b7d6b62820b" width="300">
</p>

# FramePack

Official implementation and desktop software for ["Packing Input Frame Context in Next-Frame Prediction Models for Video Generation"](https://lllyasviel.github.io/frame_pack_gitpage/).

Links: [**Paper**](https://lllyasviel.github.io/frame_pack_gitpage/pack.pdf), [**Project Page**](https://lllyasviel.github.io/frame_pack_gitpage/)

FramePack is a next-frame (next-frame-section) prediction neural network structure that generates videos progressively. 

FramePack compresses input contexts to a constant length so that the generation workload is invariant to video length.

FramePack can process a very large number of frames with 13B models even on laptop GPUs.

FramePack can be trained with a much larger batch size, similar to the batch size for image diffusion training.

**Video diffusion, but feels like image diffusion.**

# Requirements

Note that this repo is a functional desktop software with minimal standalone high-quality sampling system and memory management.

**Start with this repo before you try anything else!**

Requirements:

* Nvidia GPU in RTX 30XX, 40XX, 50XX series that supports fp16 and bf16. The GTX 10XX/20XX are not tested.
* Linux or Windows operating system.
* At least 6GB GPU memory.

To generate 1-minute video (60 seconds) at 30fps (1800 frames) using 13B model, the minimal required GPU memory is 6GB. (Yes 6 GB, not a typo. Laptop GPUs are okay.)

About speed, on my RTX 4090 desktop it generates at a speed of 2.5 seconds/frame (unoptimized) or 1.5 seconds/frame (teacache). On my laptops like 3070ti laptop or 3060 laptop, it is about 4x to 8x slower.

In any case, you will directly see the generated frames since it is next-frame(-section) prediction. So you will get lots of visual feedback before the entire video is generated.

# Installation

**Windows**:

[>>> Click Here to Download One-Click Package (CUDA 12.6 + Pytorch 2.6) <<<](https://github.com/lllyasviel/FramePack/releases/download/windows/framepack_cu126_torch26.7z)

After you download, you uncompress, use `update.bat` to update, and use `run.bat` to run.

Note that running `update.bat` is important, otherwise you may be using a previous version with potential bugs unfixed.

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/c49bd60d-82bd-4086-9859-88d472582b94)

Note that the models will be downloaded automatically. You will download more than 30GB from HuggingFace.

**Linux**:

We recommend having an independent Python 3.10.

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt

To start the GUI, run:

    python demo_gradio.py

Note that it supports `--share`, `--port`, `--server`, and so on.

The software supports PyTorch attention, xformers, flash-attn, sage-attention. By default, it will just use PyTorch attention. You can install those attention kernels if you know how. 

For example, to install sage-attention (linux):

    pip install sageattention==1.0.6

However, you are highly recommended to first try without sage-attention since it will influence results, though the influence is minimal.

# GUI

![ui](https://github.com/user-attachments/assets/8c5cdbb1-b80c-4b7e-ac27-83834ac24cc4)

On the left you upload an image and write a prompt.

On the right are the generated videos and latent previews.

Because this is a next-frame-section prediction model, videos will be generated longer and longer.

You will see the progress bar for each section and the latent preview for the next section.

Note that the initial progress may be slower than later diffusion as the device may need some warmup.

# Sanity Check

Before trying your own inputs, we highly recommend going through the sanity check to find out if any hardware or software went wrong. 

Next-frame-section prediction models are very sensitive to subtle differences in noise and hardware. Usually, people will get slightly different results on different devices, but the results should look overall similar. In some cases, if possible, you'll get exactly the same results.

## Image-to-5-seconds

Download this image:

<img src="https://github.com/user-attachments/assets/f3bc35cf-656a-4c9c-a83a-bbab24858b09" width="150">

Copy this prompt:

`The man dances energetically, leaping mid-air with fluid arm swings and quick footwork.`

Set like this:

(all default parameters, with teacache turned off)
![image](https://github.com/user-attachments/assets/0071fbb6-600c-4e0f-adc9-31980d540e9d)

The result will be:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/bc74f039-2b14-4260-a30b-ceacf611a185" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

**Important Note:**

Again, this is a next-frame-section prediction model. This means you will generate videos frame-by-frame or section-by-section.

**If you get a much shorter video in the UI, like a video with only 1 second, then it is totally expected.** You just need to wait. More sections will be generated to complete the video.

## Know the influence of TeaCache and Quantization

Download this image:

<img src="https://github.com/user-attachments/assets/42293e30-bdd4-456d-895c-8fedff71be04" width="150">

Copy this prompt:

`The girl dances gracefully, with clear movements, full of charm.`

Set like this:

![image](https://github.com/user-attachments/assets/4274207d-5180-4824-a552-d0d801933435)

Turn off teacache:

![image](https://github.com/user-attachments/assets/53b309fb-667b-4aa8-96a1-f129c7a09ca6)

You will get this:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/04ab527b-6da1-4726-9210-a8853dda5577" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

Now turn on teacache:

![image](https://github.com/user-attachments/assets/16ad047b-fbcc-4091-83dc-d46bea40708c)

About 30% users will get this (the other 70% will get other random results depending on their hardware):

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/149fb486-9ccc-4a48-b1f0-326253051e9b" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>A typical worse result.</em>
    </td>
  </tr>
</table>

So you can see that teacache is not really lossless and sometimes can influence the result a lot.

We recommend using teacache to try ideas and then using the full diffusion process to get high-quality results.

This recommendation also applies to sage-attention, bnb quant, gguf, etc., etc.

## Image-to-1-minute

<img src="https://github.com/user-attachments/assets/820af6ca-3c2e-4bbc-afe8-9a9be1994ff5" width="150">

`The girl dances gracefully, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/8c34fcb2-288a-44b3-a33d-9d2324e30cbd)

Set video length to 60 seconds:

![image](https://github.com/user-attachments/assets/5595a7ea-f74e-445e-ad5f-3fb5b4b21bee)

If everything is in order you will get some result like this eventually.

60s version:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/c3be4bde-2e33-4fd4-b76d-289a036d3a47" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

6s version:

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/37fe2c33-cb03-41e8-acca-920ab3e34861" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

# More Examples

Many more examples are in [**Project Page**](https://lllyasviel.github.io/frame_pack_gitpage/).

Below are some more examples that you may be interested in reproducing.

---

<img src="https://github.com/user-attachments/assets/99f4d281-28ad-44f5-8700-aa7a4e5638fa" width="150">

`The girl dances gracefully, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/0e98bfca-1d91-4b1d-b30f-4236b517c35e)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/cebe178a-09ce-4b7a-8f3c-060332f4dab1" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/853f4f40-2956-472f-aa7a-fa50da03ed92" width="150">

`The girl suddenly took out a sign that said "cute" using right hand`

![image](https://github.com/user-attachments/assets/d51180e4-5537-4e25-a6c6-faecae28648a)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/116069d2-7499-4f38-ada7-8f85517d1fbb" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/6d87c53f-81b2-4108-a704-697164ae2e81" width="150">

`The girl skateboarding, repeating the endless spinning and dancing and jumping on a skateboard, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/c2cfa835-b8e6-4c28-97f8-88f42da1ffdf)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/d9e3534a-eb17-4af2-a8ed-8e692e9993d2" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/6e95d1a5-9674-4c9a-97a9-ddf704159b79" width="150">

`The girl dances gracefully, with clear movements, full of charm.`

![image](https://github.com/user-attachments/assets/7412802a-ce44-4188-b1a4-cfe19f9c9118)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/e1b3279e-e30d-4d32-b55f-2fb1d37c81d2" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/90fc6d7e-8f6b-4f8c-a5df-ee5b1c8b63c9" width="150">

`The man dances flamboyantly, swinging his hips and striking bold poses with dramatic flair.`

![image](https://github.com/user-attachments/assets/1dcf10a3-9747-4e77-a269-03a9379dd9af)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/aaa4481b-7bf8-4c64-bc32-909659767115" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/62ecf987-ec0c-401d-b3c9-be9ffe84ee5b" width="150">

`The woman dances elegantly among the blossoms, spinning slowly with flowing sleeves and graceful hand movements.`

![image](https://github.com/user-attachments/assets/396f06bc-e399-4ac3-9766-8a42d4f8d383)


<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/f23f2f37-c9b8-45d5-a1be-7c87bd4b41cf" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

<img src="https://github.com/user-attachments/assets/4f740c1a-2d2f-40a6-9613-d6fe64c428aa" width="150">

`The young man writes intensely, flipping papers and adjusting his glasses with swift, focused movements.`

![image](https://github.com/user-attachments/assets/c4513c4b-997a-429b-b092-bb275a37b719)

<table>
  <tr>
    <td align="center" width="300">
      <video 
        src="https://github.com/user-attachments/assets/62e9910e-aea6-4b2b-9333-2e727bccfc64" 
        controls 
        style="max-width:100%;">
      </video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Video may be compressed by GitHub</em>
    </td>
  </tr>
</table>

---

# Project Structure

Below is the structure of the `diffusers_helper` module that contains the core components of FramePack:

```
diffusers_helper/
├── bucket_tools.py        # Tools for bucket operations
├── clip_vision.py         # CLIP vision model functionality
├── dit_common.py          # Common DiT (Diffusion Transformer) components
├── hf_login.py            # HuggingFace login utilities
├── hunyuan.py             # HunYuan model implementation
├── memory.py              # Memory management tools
├── thread_utils.py        # Utilities for threading and parallel processing
├── utils.py               # General utility functions
├── gradio/
│   └── progress_bar.py    # Progress bar implementation for Gradio UI
├── k_diffusion/
│   ├── uni_pc_fm.py       # UniPC FM sampler implementation
│   └── wrapper.py         # K-diffusion wrapper
├── models/
│   └── hunyuan_video_packed.py # FramePack video generation model
└── pipelines/
    └── k_diffusion_hunyuan.py  # K-diffusion pipeline for HunYuan

```

Files in the `diffusers_helper` directory with links to key implementations:

- [bucket_tools.py](diffusers_helper/bucket_tools.py): Tools for bucket operations
- [clip_vision.py](diffusers_helper/clip_vision.py): CLIP vision model functionality
- [dit_common.py](diffusers_helper/dit_common.py#L7): Common DiT (Diffusion Transformer) components with LayerNorm
- [hf_login.py](diffusers_helper/hf_login.py): HuggingFace login utilities
- [hunyuan.py](diffusers_helper/hunyuan.py#L8): HunYuan model implementation with encode_prompt_conds
- [memory.py](diffusers_helper/memory.py#L15): Memory management tools for efficient GPU memory usage
- [thread_utils.py](diffusers_helper/thread_utils.py#L9): Utilities for threading and parallel processing
- [utils.py](diffusers_helper/utils.py): General utility functions

Key Implementation Files:
- **models/**
  - [hunyuan_video_packed.py](diffusers_helper/models/hunyuan_video_packed.py#L723): **Main FramePack model** - `HunyuanVideoTransformer3DModelPacked` class
  - [hunyuan_video_packed.py](diffusers_helper/models/hunyuan_video_packed.py#L835): Frame packing mechanism - `process_input_hidden_states()` method
  - [hunyuan_video_packed.py](diffusers_helper/models/hunyuan_video_packed.py#L818): TeaCache optimization - `initialize_teacache()` method
  - [hunyuan_video_packed.py](diffusers_helper/models/hunyuan_video_packed.py#L604): Transformer block implementation - `HunyuanVideoTransformerBlock` class

- **pipelines/**
  - [k_diffusion_hunyuan.py](diffusers_helper/pipelines/k_diffusion_hunyuan.py#L57): K-diffusion sampling function - `sample_hunyuan()`

- **k_diffusion/**
  - [uni_pc_fm.py](diffusers_helper/k_diffusion/uni_pc_fm.py#L36): UniPC FM sampling algorithm
  - [wrapper.py](diffusers_helper/k_diffusion/wrapper.py#L5): K-diffusion wrapper for FramePack model

- **gradio/**
  - [progress_bar.py](diffusers_helper/gradio/progress_bar.py): Progress bar implementation for Gradio UI

# FramePack Model Architecture

Below is a class and function diagram of the core `hunyuan_video_packed.py` implementation, which contains the main model architecture for FramePack:

```
HunyuanVideoTransformer3DModelPacked
├── Core Architecture
│   ├── x_embedder (HunyuanVideoPatchEmbed)
│   ├── context_embedder (HunyuanVideoTokenRefiner)
│   ├── time_text_embed (CombinedTimestepGuidanceTextProjEmbeddings)
│   ├── rope (HunyuanVideoRotaryPosEmbed)
│   ├── transformer_blocks [HunyuanVideoTransformerBlock x num_layers]
│   ├── single_transformer_blocks [HunyuanVideoSingleTransformerBlock x num_single_layers]
│   ├── norm_out (AdaLayerNormContinuous)
│   └── proj_out (Linear)
│
├── Key Methods
│   ├── process_input_hidden_states() - Handles multi-resolution frame packing
│   ├── forward() - Main forward pass with context management
│   ├── initialize_teacache() - Sets up memory optimization
│   └── gradient_checkpointing_method() - Manages efficient gradient computation
│
├── Transformer Blocks
│   ├── HunyuanVideoTransformerBlock
│   │   ├── norm1 (AdaLayerNormZero)
│   │   ├── attn (Attention with HunyuanAttnProcessorFlashAttnDouble)
│   │   └── ff (FeedForward)
│   │
│   └── HunyuanVideoSingleTransformerBlock
│       ├── norm (AdaLayerNormZeroSingle)
│       ├── attn (Attention with HunyuanAttnProcessorFlashAttnSingle)
│       └── proj_out (Linear)
│
└── Utility Functions
    ├── pad_for_3d_conv() - Handles padding for 3D convolutions
    ├── center_down_sample_3d() - Downsampling for multi-resolution
    ├── get_cu_seqlens() - Manages sequence lengths
    ├── apply_rotary_emb_transposed() - Applies rotary embeddings
    └── attn_varlen_func() - Variable length attention function
```

## Clickable Links to Key Components

### Main Class
- [HunyuanVideoTransformer3DModelPacked](diffusers_helper/models/hunyuan_video_packed.py#L723)

### Core Architecture Components
- [HunyuanVideoPatchEmbed](diffusers_helper/models/hunyuan_video_packed.py#L690)
- [HunyuanVideoTokenRefiner](diffusers_helper/models/hunyuan_video_packed.py#L372)
- [CombinedTimestepGuidanceTextProjEmbeddings](diffusers_helper/models/hunyuan_video_packed.py#L215)
- [HunyuanVideoRotaryPosEmbed](diffusers_helper/models/hunyuan_video_packed.py#L421)
- [AdaLayerNormContinuous](diffusers_helper/models/hunyuan_video_packed.py#L504)

### Key Methods
- [process_input_hidden_states()](diffusers_helper/models/hunyuan_video_packed.py#L835) - Frame packing mechanism
- [forward()](diffusers_helper/models/hunyuan_video_packed.py#L894) - Main model forward pass
- [initialize_teacache()](diffusers_helper/models/hunyuan_video_packed.py#L818) - TeaCache optimization
- [gradient_checkpointing_method()](diffusers_helper/models/hunyuan_video_packed.py#L828)

### Transformer Blocks
- [HunyuanVideoTransformerBlock](diffusers_helper/models/hunyuan_video_packed.py#L604)
  - [AdaLayerNormZero](diffusers_helper/models/hunyuan_video_packed.py#L459)
  - [HunyuanAttnProcessorFlashAttnDouble](diffusers_helper/models/hunyuan_video_packed.py#L139)
- [HunyuanVideoSingleTransformerBlock](diffusers_helper/models/hunyuan_video_packed.py#L530)
  - [AdaLayerNormZeroSingle](diffusers_helper/models/hunyuan_video_packed.py#L481)
  - [HunyuanAttnProcessorFlashAttnSingle](diffusers_helper/models/hunyuan_video_packed.py#L185)

### Utility Functions
- [pad_for_3d_conv()](diffusers_helper/models/hunyuan_video_packed.py#L64)
- [center_down_sample_3d()](diffusers_helper/models/hunyuan_video_packed.py#L73)
- [get_cu_seqlens()](diffusers_helper/models/hunyuan_video_packed.py#L82)
- [apply_rotary_emb_transposed()](diffusers_helper/models/hunyuan_video_packed.py#L99)
- [attn_varlen_func()](diffusers_helper/models/hunyuan_video_packed.py#L108)

The key innovation in this architecture is the frame packing mechanism implemented in [`process_input_hidden_states()`](diffusers_helper/models/hunyuan_video_packed.py#L835), which allows the model to handle arbitrary video lengths with constant memory usage. The method combines frames at different temporal resolutions (original, 2x, and 4x) to create a fixed-length context that compresses the video history.

The TeaCache optimization ([`initialize_teacache()`](diffusers_helper/models/hunyuan_video_packed.py#L818)) further reduces computational load by reusing certain computations when changes between diffusion steps are small, resulting in up to 2x speedup with minimal quality loss.

# Comparison with Original Hunyuan Implementation

Below is a comparison between the original Hunyuan video implementation and the FramePack modified version:

| Feature | Original Hunyuan | FramePack Modified |
|---------|-----------------|-------------------|
| Main Class | `HunyuanVideoTransformer3DModel` (L821) | `HunyuanVideoTransformer3DModelPacked` (L723) |
| Frame Context | Processes full video context, growing with video length | Uses fixed-length context through frame packing mechanism |
| Memory Usage | Increases with video length | Constant regardless of video length |
| Multi-resolution Support | No support for multiple resolutions | Supports multiple temporal resolutions (original, 2x, 4x downsampled) |
| TeaCache Optimization | Not available | Implemented to reuse computations between diffusion steps (L818) |
| Attention Implementation | Uses PyTorch 2.0's scaled_dot_product_attention (L119) | Optimized with multiple attention backends (Flash, xformers, SAGE) (L108-139) |
| Forward pass | Standard forward pass (L1025-1150) | Includes TeaCache branching logic (L894-971) |

## Key Implementation Differences

1. **Frame Packing Mechanism**: 
   - Original: Processes each frame in full context with no context compression, in `HunyuanVideoTransformer3DModel.forward()` (L1025-1150)
   - FramePack: Implements `process_input_hidden_states()` (L835-893) which combines frames at different temporal resolutions into a fixed-size context

2. **Memory Management**:
   - Original: No management of previous frames - context grows proportionally to video length
   - FramePack: Processes frames at different downsampling rates (1x, 2x, 4x) to maintain fixed context size in `process_input_hidden_states()` (L835-893)
   ```python
   # Key line showing multi-resolution context management
   if clean_latents_2x is not None and clean_latent_2x_indices is not None:
       # Code that packs downsampled frames at 2x resolution
   ```

3. **TeaCache Optimization**:
   - Original: No caching mechanism, full computation for each diffusion step
   - FramePack: Implemented in `initialize_teacache()` (L818-826) and used in forward pass (L949-971)
   ```python
   # TeaCache conditional path in forward()
   if self.enable_teacache:
       modulated_inp = self.transformer_blocks[0].norm1(hidden_states, emb=temb)[0]
       # Check if computation can be reused
       if not should_calc:
           hidden_states = hidden_states + self.previous_residual
   ```

4. **Attention Implementation**:
   - Original: Uses only PyTorch 2.0's scaled_dot_product_attention (L119)
   - FramePack: Implements `attn_varlen_func()` (L108-138) with support for different attention backends
   ```python
   # Multiple attention backend support
   if sageattn is not None:
       x = sageattn(q, k, v, tensor_layout='NHD')
       return x
   if flash_attn_func is not None:
       x = flash_attn_func(q, k, v)
       return x
   ```

5. **Multi-resolution Context**:
   - Original: No multi-resolution support, processes all frames at the same resolution
   - FramePack: Implements specific handling for various resolution contexts in `process_input_hidden_states()` (L835-893)
   ```python
   # Processing 4x downsampled frames
   if clean_latents_4x is not None and clean_latent_4x_indices is not None:
       # Downsampling and processing logic for 4x context
       clean_latent_4x_rope_freqs = center_down_sample_3d(clean_latent_4x_rope_freqs, (4, 4, 4))
   ```

These architectural differences allow FramePack to maintain constant memory usage regardless of video length, making it possible to generate very long videos even on resource-constrained hardware.

# Prompting Guideline

Many people would ask how to write better prompts. 

Below is a ChatGPT template that I personally often use to get prompts:

    You are an assistant that writes short, motion-focused prompts for animating images.

    When the user sends an image, respond with a single, concise prompt describing visual motion (such as human activity, moving objects, or camera movements). Focus only on how the scene could come alive and become dynamic using brief phrases.

    Larger and more dynamic motions (like dancing, jumping, running, etc.) are preferred over smaller or more subtle ones (like standing still, sitting, etc.).

    Describe subject, then motion, then other things. For example: "The girl dances gracefully, with clear movements, full of charm."

    If there is something that can dance (like a man, girl, robot, etc.), then prefer to describe it as dancing.

    Stay in a loop: one image in, one motion prompt out. Do not explain, ask questions, or generate multiple options.

You paste the instruct to ChatGPT and then feed it an image to get prompt like this:

![image](https://github.com/user-attachments/assets/586c53b9-0b8c-4c94-b1d3-d7e7c1a705c3)

*The man dances powerfully, striking sharp poses and gliding smoothly across the reflective floor.*

Usually this will give you a prompt that works well. 

You can also write prompts yourself. Concise prompts are usually preferred, for example:

*The girl dances gracefully, with clear movements, full of charm.*

*The man dances powerfully, with clear movements, full of energy.*

and so on.

# Cite

    @article{zhang2025framepack,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }
