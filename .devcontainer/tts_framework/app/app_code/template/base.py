from abc import abstractmethod
import os
from typing import Dict


class SynthesizerBase:
    _loaded = False
    _name = ""
    model = None
    flavor = None
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
        kwargs = gen_args.update(kwargs)
        return self._generate(
            text,
            progress_callback=progress_callback,
            progress_state=progress_state,
            progress=progress,
            **kwargs
        )

    @abstractmethod
    def gen_args(self) -> Dict:
        raise NotImplementedError

    def get_args(self):
        args = {f"{self._name}_{k}": v for k, v in zip(self.gen_args()[:2])}
        return args

    def properties(self, name: str, container):
        self._name = name
        for key, value, label, _type, kwargs in zip(self.gen_args()):
            getattr(self, _type)(label, value=value, key=f"{name}_{key}", **kwargs)
        return container, self.get_args()
