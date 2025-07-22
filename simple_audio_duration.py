import torch
import numpy as np

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
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "mix_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mixed_audio",)
    FUNCTION = "overlay_audio"
    CATEGORY = "audio/mixing"
    
    def overlay_audio(self, audio1, audio2, mix_ratio):
        """Mix two audio tracks together"""
        waveform1 = audio1["waveform"]
        waveform2 = audio2["waveform"]
        sample_rate1 = audio1["sample_rate"]
        sample_rate2 = audio2["sample_rate"]
        
        # Ensure sample rates match
        if sample_rate1 != sample_rate2:
            raise ValueError(f"Sample rates must match. Audio1: {sample_rate1}Hz, Audio2: {sample_rate2}Hz")
        
        # Get the length of the longer audio
        max_length = max(waveform1.shape[-1], waveform2.shape[-1])
        
        # Ensure both waveforms have the same number of channels
        if waveform1.shape[0] != waveform2.shape[0]:
            # Convert mono to stereo if needed
            if waveform1.shape[0] == 1 and waveform2.shape[0] == 2:
                waveform1 = waveform1.repeat(2, 1)
            elif waveform1.shape[0] == 2 and waveform2.shape[0] == 1:
                waveform2 = waveform2.repeat(2, 1)
            else:
                raise ValueError(f"Incompatible channel counts: {waveform1.shape[0]} and {waveform2.shape[0]}")
        
        # Pad shorter audio with zeros to match the longer one
        if waveform1.shape[-1] < max_length:
            padding = max_length - waveform1.shape[-1]
            waveform1 = torch.nn.functional.pad(waveform1, (0, padding))
        
        if waveform2.shape[-1] < max_length:
            padding = max_length - waveform2.shape[-1]
            waveform2 = torch.nn.functional.pad(waveform2, (0, padding))
        
        # Mix the audio tracks
        # mix_ratio of 0.5 means equal mix
        # mix_ratio of 0.0 means only audio2
        # mix_ratio of 1.0 means only audio1
        mixed_waveform = (waveform1 * mix_ratio) + (waveform2 * (1.0 - mix_ratio))
        
        # Prevent clipping by normalizing if necessary
        max_val = torch.abs(mixed_waveform).max()
        if max_val > 1.0:
            mixed_waveform = mixed_waveform / max_val
        
        return ({
            "waveform": mixed_waveform,
            "sample_rate": sample_rate1
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
