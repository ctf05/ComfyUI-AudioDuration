import torch
import numpy as np
import torch.nn.functional as F

class SimpleAudioDuration:
    """
    A simple node to get the duration of an audio file in seconds
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("duration_seconds",)
    FUNCTION = "get_duration"
    CATEGORY = "audio/analysis"
    
    def get_duration(self, audio):
        """Calculate audio duration in seconds"""
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Calculate duration: number of samples / sample rate
        duration = float(waveform.shape[-1]) / float(sample_rate)
        
        return (duration,)

class SimpleAudioOverlay:
    """
    Overlay two audio tracks starting from the beginning (mix them together)
    Supports different sample rates by resampling
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "blend_mode": (["add", "average", "mix"], {
                    "default": "add"
                }),
                "mix_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "target_sample_rate": (["auto", "audio1", "audio2", "higher", "lower"], {
                    "default": "higher"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mixed_audio",)
    FUNCTION = "overlay_audio"
    CATEGORY = "audio/mixing"
    
    def resample_audio(self, waveform, orig_sr, target_sr):
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return waveform
        
        # Calculate new length
        duration = waveform.shape[-1] / orig_sr
        new_length = int(duration * target_sr)
        
        # Handle different input shapes
        original_shape = waveform.shape
        
        # If waveform already has batch dimension (3D), use it as is
        if waveform.dim() == 3:
            # Shape is [batch, channels, samples]
            batch_waveform = waveform
        elif waveform.dim() == 2:
            # Shape is [channels, samples], add batch dimension
            batch_waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 1:
            # Shape is [samples], add batch and channel dimensions
            batch_waveform = waveform.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected waveform dimensions: {waveform.dim()}")
        
        # Resample
        resampled = F.interpolate(
            batch_waveform,
            size=new_length,
            mode='linear',
            align_corners=False
        )
        
        # Return to original shape format
        if waveform.dim() == 3:
            return resampled
        elif waveform.dim() == 2:
            return resampled.squeeze(0)
        else:  # dim == 1
            return resampled.squeeze(0).squeeze(0)
    
    def overlay_audio(self, audio1, audio2, blend_mode, mix_ratio, target_sample_rate):
        """Mix two audio tracks together with sample rate conversion"""
        waveform1 = audio1["waveform"]
        waveform2 = audio2["waveform"]
        sample_rate1 = audio1["sample_rate"]
        sample_rate2 = audio2["sample_rate"]
        
        # Determine target sample rate
        if target_sample_rate == "auto" or target_sample_rate == "higher":
            final_sample_rate = max(sample_rate1, sample_rate2)
        elif target_sample_rate == "lower":
            final_sample_rate = min(sample_rate1, sample_rate2)
        elif target_sample_rate == "audio1":
            final_sample_rate = sample_rate1
        elif target_sample_rate == "audio2":
            final_sample_rate = sample_rate2
        
        # Resample if necessary
        if sample_rate1 != final_sample_rate:
            waveform1 = self.resample_audio(waveform1, sample_rate1, final_sample_rate)
        
        if sample_rate2 != final_sample_rate:
            waveform2 = self.resample_audio(waveform2, sample_rate2, final_sample_rate)
        
        # Get the length of the longer audio
        max_length = max(waveform1.shape[-1], waveform2.shape[-1])
        
        # Handle batch dimensions if present
        has_batch1 = waveform1.dim() == 3
        has_batch2 = waveform2.dim() == 3
        
        # Remove batch dimension for processing if present
        if has_batch1:
            waveform1 = waveform1.squeeze(0)
        if has_batch2:
            waveform2 = waveform2.squeeze(0)
        
        # Now waveforms should be 2D [channels, samples] or 1D [samples]
        # Get the actual channel count
        channels1 = waveform1.shape[0] if waveform1.dim() == 2 else 1
        channels2 = waveform2.shape[0] if waveform2.dim() == 2 else 1
        
        # Ensure both waveforms have the same number of channels
        if channels1 != channels2:
            # Convert mono to stereo if needed
            if channels1 == 1 and channels2 == 2:
                if waveform1.dim() == 1:
                    waveform1 = waveform1.unsqueeze(0).repeat(2, 1)
                else:
                    waveform1 = waveform1.repeat(2, 1)
            elif channels1 == 2 and channels2 == 1:
                if waveform2.dim() == 1:
                    waveform2 = waveform2.unsqueeze(0).repeat(2, 1)
                else:
                    waveform2 = waveform2.repeat(2, 1)
            else:
                raise ValueError(f"Incompatible channel counts: {channels1} and {channels2}")
        
        # Ensure both are 2D now
        if waveform1.dim() == 1:
            waveform1 = waveform1.unsqueeze(0)
        if waveform2.dim() == 1:
            waveform2 = waveform2.unsqueeze(0)
        
        # Pad shorter audio with zeros to match the longer one
        if waveform1.shape[-1] < max_length:
            padding = max_length - waveform1.shape[-1]
            waveform1 = F.pad(waveform1, (0, padding))
        
        if waveform2.shape[-1] < max_length:
            padding = max_length - waveform2.shape[-1]
            waveform2 = F.pad(waveform2, (0, padding))
        
        # Mix based on blend mode
        if blend_mode == "add":
            # Simple addition - sounds like both playing at once
            mixed_waveform = waveform1 + waveform2
            
        elif blend_mode == "average":
            # Average - each at 50% volume
            mixed_waveform = (waveform1 + waveform2) / 2.0
            
        elif blend_mode == "mix":
            # Custom mix based on ratio
            # mix_ratio of 0.5 means equal mix
            # mix_ratio of 0.0 means only audio2
            # mix_ratio of 1.0 means only audio1
            mixed_waveform = (waveform1 * mix_ratio) + (waveform2 * (1.0 - mix_ratio))
        
        # Prevent clipping by normalizing if necessary
        max_val = torch.abs(mixed_waveform).max()
        if max_val > 1.0:
            mixed_waveform = mixed_waveform / max_val
        
        # Add batch dimension back if either input had it
        if has_batch1 or has_batch2:
            mixed_waveform = mixed_waveform.unsqueeze(0)
        
        return ({
            "waveform": mixed_waveform,
            "sample_rate": final_sample_rate
        },)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SimpleAudioDuration": SimpleAudioDuration,
    "SimpleAudioOverlay": SimpleAudioOverlay
}

# Optional: Nice display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleAudioDuration": "Audio Duration",
    "SimpleAudioOverlay": "Audio Overlay (Mix)"
}
