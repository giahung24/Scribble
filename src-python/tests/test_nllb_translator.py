from stt_providers.ondevice.translator import NllbTranslator

class _FakeTokenizer:
    def __init__(self): self.src_lang = None
    def convert_ids_to_tokens(self, ids): return [f"tok{i}" for i in ids]
    def encode(self, text): return [1, 2, 3]
    def convert_tokens_to_ids(self, toks): return [9, 9]
    def decode(self, ids, skip_special_tokens=True): return "translated"

class _FakeCT2:
    def __init__(self): self.calls = []
    def translate_batch(self, source, target_prefix=None, **kw):
        self.calls.append((source, target_prefix))
        class _R: hypotheses = [["<tgt>", "tok9", "tok9"]]
        return [_R()]

def test_translate_passes_target_prefix_and_decodes():
    tok, ct2 = _FakeTokenizer(), _FakeCT2()
    tr = NllbTranslator(translator=ct2, tokenizer=tok)
    out = tr.translate("xin chào", src="vie_Latn", tgt="eng_Latn")
    assert out == "translated"
    # target language must be passed as the decoder target prefix
    assert ct2.calls[0][1] == [["eng_Latn"]]
    # src_lang set on the tokenizer before encoding
    assert tok.src_lang == "vie_Latn"
