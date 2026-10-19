"""Крошечный случайный BERT + WordPiece-токенизатор для smoke-теста без доступа к Hugging Face."""
import sys
from pathlib import Path

import pandas as pd
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

corpus, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
texts = pd.read_csv(corpus)["text"].str.lower().tolist()

wp = BertWordPieceTokenizer(lowercase=True)
wp.train_from_iterator(texts, vocab_size=2000, min_frequency=1)
wp.save_model(str(out))
tokenizer = BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=True)

config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=128, max_position_embeddings=128, num_labels=2)
BertForSequenceClassification(config).save_pretrained(out)
tokenizer.save_pretrained(out)
print(f"tiny model → {out}")
