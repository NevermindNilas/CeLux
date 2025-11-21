import torch

def yuv_to_rgb(yuv_tensor: torch.Tensor, width: int, height: int, bit_depth: int = 8) -> torch.Tensor:
    """
    Convert a flat YUV420P tensor to an RGB Float32 tensor (H, W, 3).
    Preserves super-blacks (values < 16) as negative values.
    
    Args:
        yuv_tensor: 1D tensor containing Y, U, V planes.
        width: Width of the frame.
        height: Height of the frame.
        bit_depth: Bit depth of the source (8, 10, 12, etc.).
        
    Returns:
        RGB tensor of shape (H, W, 3) in Float32.
    """
    
    # Calculate plane sizes
    y_size = width * height
    uv_width = width // 2
    uv_height = height // 2
    uv_size = uv_width * uv_height
    
    # Check if tensor size matches expected size
    expected_size = y_size + 2 * uv_size
    if yuv_tensor.numel() < expected_size:
        raise ValueError(f"Tensor too small. Expected {expected_size}, got {yuv_tensor.numel()}")
        
    # Slice planes
    y_plane = yuv_tensor[:y_size].view(height, width)
    u_plane = yuv_tensor[y_size : y_size + uv_size].view(uv_height, uv_width)
    v_plane = yuv_tensor[y_size + uv_size : y_size + 2 * uv_size].view(uv_height, uv_width)
    
    # Convert to float
    y = y_plane.float()
    u = u_plane.float()
    v = v_plane.float()
    
    # Upsample U and V to match Y size
    # Note: swscale usually uses bilinear or bicubic. We use bilinear here for speed/compatibility.
    # We need to add batch and channel dims for interpolate: (1, 1, H, W)
    u = torch.nn.functional.interpolate(u.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze()
    v = torch.nn.functional.interpolate(v.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze()
    
    # Constants for BT.709 Limited Range
    # Y: 16-235, U/V: 16-240 (centered at 128)
    
    scale = 1.0
    offset_y = 16.0
    offset_uv = 128.0
    
    if bit_depth > 8:
        shift = bit_depth - 8
        scale = float(1 << shift)
        offset_y *= scale
        offset_uv *= scale
        
    # Normalize to 0-1 range (roughly) or keep in 8-bit range?
    # Standard formula works on 8-bit range usually.
    # Let's work in the native range and then normalize if needed, or just output 0-255 float.
    # The prompt implies we want to preserve values, so 0-255 float is good.
    
    y_shifted = y - offset_y
    u_shifted = u - offset_uv
    v_shifted = v - offset_uv
    
    # BT.709 conversion
    # R = Y + 1.5748 * V
    # G = Y - 0.1873 * U - 0.4681 * V
    # B = Y + 1.8556 * U
    
    r = y_shifted + 1.5748 * v_shifted
    g = y_shifted - 0.0722 * u_shifted - 0.6398 * v_shifted # Wait, checking coefficients
    b = y_shifted + 1.8556 * u_shifted
    
    # Rec. 709 (HDTV)
    # Kr = 0.2126, Kb = 0.0722
    # R = Y + 2*(1-Kr)*Cr
    # B = Y + 2*(1-Kb)*Cb
    # G = Y - ...
    
    # Using standard integer approx coefficients converted to float:
    # R = 1.164 * (Y-16) + 1.793 * (V-128)
    # This includes the range expansion (255/219).
    
    # If we want "Raw" conversion without range expansion (preserving 16-235 as 0-1? No).
    # The prompt says: "Standard RGB conversion formula: R = 1.164 * (Y - 16) + ..."
    # This implies range expansion (Limited to Full).
    
    # 1.164 = 255 / 219.
    
    # Let's use the standard Limited->Full conversion matrix for BT.709.
    
    # Y_coeff = 1.16438
    # R = Y_coeff * (Y - 16) + 1.79274 * (V - 128)
    # G = Y_coeff * (Y - 16) - 0.21325 * (U - 128) - 0.53291 * (V - 128)
    # B = Y_coeff * (Y - 16) + 2.11240 * (U - 128)
    
    y_term = 1.16438 * y_shifted
    
    r = y_term + 1.79274 * v_shifted
    g = y_term - 0.21325 * u_shifted - 0.53291 * v_shifted
    b = y_term + 2.11240 * u_shifted
    
    # Stack to (H, W, 3)
    rgb = torch.stack([r, g, b], dim=-1)
    
    # Normalize to 0-1 if requested? Or keep 0-255?
    # Usually PyTorch float images are 0-1.
    # But if we want to match 8-bit integer values, 0-255 is easier to debug.
    # Let's return 0-255 float.
    
    if bit_depth > 8:
        # Scale back to 8-bit range for consistency?
        # Or keep in 10-bit range (0-1023)?
        # If we return float, 0-1 is best.
        # But the user example showed "Y=2 maps to RGB=2". This implies 0-255 range.
        pass
        
    return rgb
