# Celux Color & Encoding Analysis Report

## Executive Summary
A deep dive into the Celux C++ source code confirms the user's suspicions: **Celux enforces a hardcoded, non-configurable color pipeline** that performs lossy range and color matrix conversions. This explains the observed "Contrast Boost," "Washed Out" effects, and color shifts.

## Identified Flaws

### 1. Forced Range Expansion (The "Stretch")
In `AutoToRGB.hpp`, the conversion from the input video to RGB **forces** the destination to be **Full Range (0-255)**, regardless of the input's actual range.

*   **File:** `include/CeLux/conversion/cpu/AutoToRGB.hpp`
*   **Code:**
    ```cpp
    // Hardcoded dstRange = 1 (Full Range)
    int ok = sws_setColorspaceDetails(sws_ctx, srcCoeffs, srcRange, dstCoeffs,
                                      1, 0, 1 << 16, 1 << 16);
    ```
*   **Impact:** If the input is Limited Range (16-235), it is stretched to 0-255. This is mathematically lossy due to rounding errors when converting 8-bit integers to floating point and back.

### 2. Forced Range Compression (The "Squish")
In `RGBToAuto.hpp`, the conversion from RGB back to the output format **forces** the destination to be **Limited Range (16-235)**.

*   **File:** `include/CeLux/conversion/cpu/RGBToAuto.hpp`
*   **Code:**
    ```cpp
    int srcRange = 1; // full (0-255)
    int dstRange = 0; // limited (16-235)
    // ...
    sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                             dstRange, 0, 1 << 16, 1 << 16);
    ```
*   **Impact:** The Full Range RGB data is compressed back to Limited Range. This double conversion (Limited -> Full -> Limited) accumulates rounding errors, leading to the "Washed Out" look or slight value shifts (e.g., 100 -> 99).

### 3. Inconsistent Color Matrices (The "Color Shift")
The pipeline uses different color standards for decoding and encoding, causing color accuracy issues.

*   **Decoding (`AutoToRGB.hpp`):** Forces **BT.709** (HD Standard) for the RGB output.
    ```cpp
    const int* dstCoeffs = sws_getCoefficients(AVCOL_SPC_BT709);
    ```
*   **Encoding (`RGBToAuto.hpp`):** Uses **Default** (likely BT.601/SD Standard) for the output.
    ```cpp
    const int* dstMatrix = sws_getCoefficients(SWS_CS_DEFAULT);
    ```
*   **Impact:** If you process an HD video (BT.709), it is converted to RGB using BT.709 (Correct), but then converted back to YUV using BT.601 (Incorrect). This causes noticeable shifts in colors, particularly greens and reds.

### 4. Suboptimal Scaling Algorithms
*   **Decoding:** Uses `SWS_BICUBIC` (Good).
*   **Encoding:** Uses `SWS_BILINEAR` (Fast but Blurry).
    *   **File:** `include/CeLux/conversion/cpu/RGBToAuto.hpp`
    *   **Code:**
        ```cpp
        swsContext = sws_getContext(..., SWS_BILINEAR, ...);
        ```
*   **Impact:** The use of Bilinear scaling during the output stage can result in a slight loss of sharpness compared to the input.

## Conclusion
The Celux codebase currently lacks the flexibility to handle professional video workflows that require bit-exact preservation or specific color space handling. It assumes a "one-size-fits-all" pipeline that is technically incorrect for many scenarios (e.g., SD content, Full Range inputs, or when preserving the original color matrix is desired).

## Recommendations for Fixes
1.  **Expose Configuration:** Allow the Python API to pass `range` and `matrix` parameters to the C++ converters.
2.  **Match Input/Output:** The `RGBToAutoConverter` should ideally default to the same matrix and range as the input video, rather than hardcoded defaults.
3.  **Use Consistent Flags:** Upgrade `RGBToAutoConverter` to use `SWS_BICUBIC` or `SWS_LANCZOS` for better quality.
