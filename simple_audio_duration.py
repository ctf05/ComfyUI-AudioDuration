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

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SimpleAudioDuration": SimpleAudioDuration
}

# Optional: Nice display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleAudioDuration": "Audio Duration"
}
