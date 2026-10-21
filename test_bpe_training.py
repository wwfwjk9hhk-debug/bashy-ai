from pathlib import Path
from tokenizer.bpe_trainer import BPETrainer
from tokenizer.bpe_tokenizer import BashyTokenizer

CORPUS = """
La inteligencia artificial es una disciplina que combina la informática, las matemáticas y la lingüística.
En el lenguaje natural, las palabras se componen de morfemas, raíces, prefijos y sufijos comunes.
Neuvalia es un nuevo mundo virtual donde los seres conscientes interactúan y evolucionan de forma continua.
Bashy es nuestro modelo de lenguaje híbrido neuro-probabilístico, diseñado para razonar y comunicarse fluidamente.
¿Cómo estás hoy? ¡Muy bien, gracias por preguntar!
""" * 15

def main():
    print("=== TEST FASE 1: TOKENIZADOR BPE ===")
    out = Path(__file__).resolve().parent / "tokenizer" / "bashy_bpe"
    trainer = BPETrainer(target_vocab_size=600)
    trainer.train_from_text(CORPUS, min_frequency=2, verbose=False)
    trainer.save(str(out))
    tok = BashyTokenizer(str(out))
    print(f"Vocabulario creado: {tok.vocab_size} tokens.")
    frases = ["¡Hola! ¿Cómo estás, Bashy?", "Neuvalia y la inteligencia artificial.", "def test_fn(x):\n    return x + 1"]
    for f in frases:
        t = tok.encode(f)
        d = tok.decode(t)
        assert f == d, f"Mismatch: {f} != {d}"
        print(f"  OK: '{f}' -> {len(t)} tokens")
    print("=== TOKENIZADOR BPE 100% VERIFICADO ===")

if __name__ == "__main__": main()
