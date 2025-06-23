import re
from typing import List, Tuple
from dataclasses import replace
from misaki import en, espeak, ja, zh

from misaki.token import MToken
from misaki.en import (
    merge_tokens,
    LINK_REGEX,
    is_digit,
    TokenContext,
    SUBTOKEN_JUNKS
)


class LocalG2P(en.G2P):
    @staticmethod
    def preprocess(text):
        result = ''
        tokens = []
        features = {}
        last_end = 0
        text = text.lstrip()
        for m in LINK_REGEX.finditer(text):
            result += text[last_end:m.start()]
            tokens.extend(text[last_end:m.start()].split())
            f = m.group(2)
            if is_digit(f[1 if f[:1] in ('-', '+') else 0:]):
                f = int(f)
            elif f in ('0.5', '+0.5'):
                f = 0.5
            elif f == '-0.5':
                f = -0.5
            elif len(f) > 1 and f[0] == '/' and f[-1] == '/':
                f = f[0] + f[1:].rstrip('/')
            elif len(f) > 1 and f[0] == '#' and f[-1] == '#':
                f = f[0] + f[1:].rstrip('#')
            else:
                f = None
            if f is not None:
                features[len(tokens)] = f
            result += m.group(1)
            tokens.append(m.group(1))
            last_end = m.end()
        if last_end < len(text):
            result += text[last_end:]
            tokens.extend(text[last_end:].split())
        return result, tokens, features

    def __call__(self, text: str, preprocess=True) -> Tuple[str, List[MToken]]:
        preprocess = self.preprocess if preprocess == True else preprocess
        text, tokens, features = preprocess(text) if preprocess else (text, [], {})
        print(tokens)
        tokens = self.tokenize(text, tokens, features)
        tokens = self.fold_left(tokens)
        tokens = self.retokenize(tokens)
        print(tokens)
        # print("56", tokens)
        ctx = TokenContext()
        for i, w in reversed(list(enumerate(tokens))):
            if not isinstance(w, list):
                if w.phonemes is None:
                    w.phonemes, w.rating = self.lexicon(replace(w, _=w._), ctx)
                if w.phonemes is None and self.fallback is not None:
                    w.phonemes, w.rating = self.fallback(replace(w, _=w._))
                ctx = self.token_context(ctx, w.phonemes, w)
                continue
            left, right = 0, len(w)
            should_fallback = False
            while left < right:
                if any(tk._.alias is not None or tk.phonemes is not None for tk in w[left:right]):
                    tk = None
                else:
                    tk = merge_tokens(w[left:right])
                ps, rating = (None, None) if tk is None else self.lexicon(tk, ctx)
                if ps is not None:
                    w[left].phonemes = ps
                    w[left]._.rating = rating
                    for x in w[left+1:right]:
                        x.phonemes = ''
                        x.rating = rating
                    ctx = self.token_context(ctx, ps, tk)
                    right = left
                    left = 0
                elif left + 1 < right:
                    left += 1
                else:
                    right -= 1
                    tk = w[right]
                    if tk.phonemes is None:
                        if all(c in SUBTOKEN_JUNKS for c in tk.text):
                            tk.phonemes = ''
                            tk._.rating = 3
                        elif self.fallback is not None:
                            should_fallback = True
                            break
                    left = 0
            if should_fallback:
                tk = merge_tokens(w)
                w[0].phonemes, w[0]._.rating = self.fallback(tk)
                for j in range(1, len(w)):
                    w[j].phonemes = ''
                    w[j]._.rating = w[0]._.rating
            else:
                self.resolve_tokens(w)
        tokens = [merge_tokens(tk, unk=self.unk) if isinstance(tk, list) else tk for tk in tokens]
        if self.version != '2.0':
            for tk in tokens:
                if tk.phonemes:
                    tk.phonemes = tk.phonemes.replace('ɾ', 'T').replace('ʔ', 't')
        result = ''.join((self.unk if tk.phonemes is None else tk.phonemes) + tk.whitespace for tk in tokens)
        return result, tokens

def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts

def main():
    fallback = espeak.EspeakFallback(british=False)
    g2p = LocalG2P(trf=False, british=False, fallback=fallback, unk='')
    
    text = "How are you doing?"
    phonemes = g2p(text)
    print(text)
    print(phonemes[0])
    
    # g2p = ja.JAG2P()
    
    # print(g2p("こんにちは"))
    vocab = get_vocab()
    print(vocab)
    return


if __name__ == "__main__":
    main()
