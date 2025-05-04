from template_kokoro import KokoroModelV1ONNX
from template_musicgen import MusicGenTransformers


class SynthFactory:
    """Factory class to create appropriate TTS model version"""
    
    @staticmethod
    def create_model(model_key="kokoro/v1.0.0-onnx"):
        """Create TTS model instance for specified version
        
        Args:
            version: Model version to use ("v0.19" or "v1.0.0")
            
        Returns:
            TTSModel or TTSModelV1 instance
        """
        package, version = model_key.split("/", 1)
        versions = {}
        if package == "kokoro":
            versions = {
                "v1.0.0-onnx": KokoroModelV1ONNX
            }
        elif package == "musicgen":
            versions = {
                "small": MusicGenTransformers
            }
        else:
            raise ValueError(f"Unsupported model: {package}")
        
        package_version = versions.get(version, None)
        if package_version is None:
            raise ValueError(f"Unsupported version: {version}")
        return package_version()
