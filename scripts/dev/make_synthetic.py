"""Синтетический датасет для smoke-теста пайплайна без интернета (НЕ для оценки качества)."""
import random
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/smoke/synthetic.csv")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
rng = random.Random(0)

subjects = ["автор", "этот депутат", "судья", "тренер", "сосед", "блогер", "чиновник", "комментатор", "ты", "вы"]
topics = ["статья", "матч", "новость", "решение", "закон", "фильм", "концерт", "ремонт дороги", "погода", "цены"]
good = ["отличная", "интересная", "полезная", "спорная, но любопытная", "хорошая", "странная", "долгожданная"]
neutral_tpl = [
    "{t} {g}, спасибо", "Согласен, {t} действительно {g}", "Когда уже закончат {t}?", "{s} прав, {t} {g}",
    "А мне кажется, {t} так себе", "Интересно, что будет с {t} дальше", "Не понимаю, почему все ругают {t}",
    "Ждём продолжения, {t} {g}", "{s} молодец, {t} {g}", "Хорошо написано про {t}",
]
insults = ["дебил", "идиот", "урод", "мразь", "тварь", "дурак", "кретин", "придурок", "ублюдок", "чмо"]
profanity = ["бля", "нахуй", "пиздец", "ебаный", "хуйня", "заебал"]
implicit = ["чтоб ты провалился", "таким не место среди людей", "закрой рот и не позорься",
            "иди лечись, больной", "вали отсюда", "ты ничтожество"]
toxic_tpl = [
    "{s} {i}", "{s} {i}, {t} {p}", "{p}, {t} опять", "{s} {i} и {i2}", "{m}, {s}", "{s} {i}, {m}",
    "я тебя найду и убью", "{t} {p} {p2}", "продам права, пиши в лс", "закладки меф, телега @shop_bot",
]


def fill(tpl):
    return tpl.format(s=rng.choice(subjects), t=rng.choice(topics), g=rng.choice(good),
                      i=rng.choice(insults), i2=rng.choice(insults), p=rng.choice(profanity),
                      p2=rng.choice(profanity), m=rng.choice(implicit))


rows = [{"text": fill(rng.choice(neutral_tpl)), "label": 0} for _ in range(int(N * 0.65))]
rows += [{"text": fill(rng.choice(toxic_tpl)), "label": 1} for _ in range(N - len(rows))]
for r in rows:  # шум разметки 3% и случайные хвосты, чтобы тексты не совпадали
    if rng.random() < 0.03:
        r["label"] = 1 - r["label"]
    r["text"] += " " + str(rng.randint(0, 10**6))
OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).sample(frac=1, random_state=0).to_csv(OUT, index=False)
print(f"{OUT}: {len(rows)} строк")
