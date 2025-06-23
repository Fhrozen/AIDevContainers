

class OuteTTSModel:
    def __init__(
        self,
        config: HFModelConfig
    ) -> None:
        self.device = torch.device(
            config.device if config.device is not None
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.config = config
        self._device = config.device
        self.languages = config.languages
        self.language = config.language
        self.verbose = config.verbose

        self.audio_codec = "" #AudioCodec(self.device, config.wavtokenizer_model_path)
        self.prompt_processor = "" # PromptProcessor(config.tokenizer_path, self.languages)
        self.model = "" # HFModel(config.model_path, self.device, config.dtype, config.additional_model_config)

    def prepare_prompt(self, text: str, speaker: dict = None):
        prompt = self.prompt_processor.get_completion_prompt(text, self.language, speaker)
        return self.prompt_processor.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

    def get_audio(self, tokens):
        output = self.prompt_processor.extract_audio_from_tokens(tokens)
        if not output:
            logger.warning("No audio tokens found in the output")
            return None

        return self.audio_codec.decode(
            torch.tensor([[output]], dtype=torch.int64).to(self.audio_codec.device)
        )

    def create_speaker(
        }

    def save_speaker(self, speaker: dict, path: str):
        with open(path, "w") as f:
            json.dump(speaker, f, indent=2)

    def load_speaker(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def print_default_speakers(self):
        total_speakers = sum(len(speakers) for speakers in _DEFAULT_SPEAKERS.values())
        print("\n=== ALL AVAILABLE SPEAKERS ===")
        print(f"Total: {total_speakers} speakers across {len(_DEFAULT_SPEAKERS)} languages")
        print("-" * 50)

        for language, speakers in _DEFAULT_SPEAKERS.items():
            print(f"\n{language.upper()} ({len(speakers)} speakers):")
            for speaker_name in speakers.keys():
                print(f"  - {speaker_name}")

        print("\n\n=== SPEAKERS FOR CURRENT INTERFACE LANGUAGE ===")
        if self.language.lower() in _DEFAULT_SPEAKERS:
            current_speakers = _DEFAULT_SPEAKERS[self.language.lower()]
            print(f"Language: {self.language.upper()} ({len(current_speakers)} speakers)")
            print("-" * 50)
            for speaker_name in current_speakers.keys():
                print(f"  - {speaker_name}")
        else:
            print(f"No speakers available for current language: {self.language}")

        print("\nTo use a speaker: load_default_speaker(name)\n")

    def load_default_speaker(self, name: str):
        name = name.lower().strip()
        language = self.language.lower().strip()
        if language not in _DEFAULT_SPEAKERS:
            raise ValueError(f"Speaker for language {language} not found")

        speakers = _DEFAULT_SPEAKERS[language]
        if name not in speakers:
            raise ValueError(f"Speaker {name} not found for language {language}")

        return self.load_speaker(speakers[name])

    def change_language(self, language: str):
        language = language.lower().strip()
        if language not in self.languages:
            raise ValueError(f"Language {language} is not supported by the current model")
        self.language = language

    def check_generation_max_length(self, max_length):
        if max_length is None:
            raise ValueError("max_length must be specified.")
        if max_length > self.config.max_seq_length:
            raise ValueError(f"Requested max_length ({max_length}) exceeds the current max_seq_length ({self.config.max_seq_length}).")

    def generate(
            self,
            text: str,
            speaker: dict = None,
            temperature: float = 0.1,
            repetition_penalty: float = 1.1,
            max_length: int = 4096,
            additional_gen_config={},
        ) -> ModelOutput:
        input_ids = self.prepare_prompt(text, speaker)
        if self.verbose:
            logger.info(f"Input tokens: {input_ids.size()[-1]}")
            logger.info("Generating audio...")

        self.check_generation_max_length(max_length)

        output = self.model.generate(
            input_ids=input_ids,
            config=GenerationConfig(
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_length=max_length,
                additional_gen_config=additional_gen_config,
            )
        )
        audio = self.get_audio(output[input_ids.size()[-1]:])
        if self.verbose:
            logger.info("Audio generation completed")

        return ModelOutput(audio, self.audio_codec.sr)
