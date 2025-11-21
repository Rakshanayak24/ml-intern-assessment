from src.ngram_model import TrigramModel

def test_fit_and_generate():
    model = TrigramModel()
    text = "I am a test sentence. This is another test sentence."
    model.fit(text)
    generated = model.generate()
    assert isinstance(generated, str)
    assert len(generated.split()) > 0

def test_empty_text():
    model = TrigramModel()
    model.fit("")
    generated = model.generate()
    assert generated == ""

def test_short_text():
    model = TrigramModel()
    model.fit("I am.")
    generated = model.generate()
    assert isinstance(generated, str)



