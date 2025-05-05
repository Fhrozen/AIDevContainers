from abc import abstractmethod
import os
from typing import Dict


class SynthesizerBase:
    _loaded = False
    _name = ""
    _flavor = None
    model = None
    tokenizer = None
    device = None
    dirpath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "downloaded")
    )

    def __init__(self):
        return

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def _generate(self):
        pass

    def generate(
        self,
        text: str,
        progress_callback = None,
        progress_state = None,
        progress = None,
        **kwargs
    ):
        gen_args = self.get_args()
        for k, v in kwargs.items():
            gen_args[k] = v
        kwargs = {k: v for k, v in gen_args.items()}
        return self._generate(
            text,
            progress_callback=progress_callback,
            progress_state=progress_state,
            progress=progress,
            **kwargs
        )

    @abstractmethod
    def gen_args(self, only_values: bool = False) -> Dict:
        raise NotImplementedError

    def get_args(self):
        args = {k: v for (k, v) in zip(*self.gen_args(only_values=True))}
        return args

    def properties(self, name: str, container):
        self._name = name
        for key, value, label, _type, kwargs in zip(*self.gen_args()):
            _widget = getattr(self, _type)
            if _type == "selectbox":
                options = kwargs.pop("options")
                _widget(label, options, index=value, key=f"{name}_{key}", **kwargs)
            else:
                _widget(label, value=value, key=f"{name}_{key}", **kwargs)
        return container, self.get_args()
