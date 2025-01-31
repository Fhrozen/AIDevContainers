from tts_model import TTSModel
from tts_model_v1 import TTSModelV1

class TTSFactory:
    """Factory class to create appropriate TTS model version"""
    
    @staticmethod
    def create_model(version="v1.0.0"):
        """Create TTS model instance for specified version
        
        Args:
            version: Model version to use ("v0.19" or "v1.0.0")
            
        Returns:
            TTSModel or TTSModelV1 instance
        """
        if version == "v0.19":
            return TTSModel()
        elif version == "v1.0.0":
            return TTSModelV1()
        else:
            raise ValueError(f"Unsupported version: {version}")
        