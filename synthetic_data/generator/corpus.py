"""Curated, dependency-free knowledge banks for the offline generator.

These banks let the offline teacher produce *unique*, self-consistent text
samples for the tasks that need source material (summarization, QA,
translation, instruction topics) without any external model or corpus
download.  All content is synthesised from closed vocabularies so it stays
internally consistent and reproducible.
"""

from __future__ import annotations

import random
from typing import Dict, List

# ---------------------------------------------------------------------------
# Generic topic / entity pools
# ---------------------------------------------------------------------------

TOPICS = [
    "renewable energy", "urban gardening", "sleep hygiene", "personal finance",
    "machine learning", "world history", "nutrition", "space exploration",
    "classical music", "cybersecurity", "volunteering", "remote work",
    "wildlife conservation", "public transport", "creative writing",
    "data privacy", "cooking", "physical exercise", "mental health",
    "entrepreneurship", "photography", "astronomy", "oceanography",
    "linguistics", "architecture",
]

SUBJECTS = [
    "researchers", "engineers", "students", "farmers", "scientists",
    "designers", "teachers", "athletes", "doctors", "artists",
]
VERBS = [
    "studied", "measured", "improved", "documented", "analysed",
    "designed", "reported", "observed", "tested", "proposed",
]
OBJECTS = [
    "efficiency", "biodiversity", "productivity", "air quality",
    "recovery time", "user satisfaction", "crop yield", "safety",
    "retention", "performance",
]
PLACES = [
    "in Norway", "across Europe", "in a coastal city", "on a rural campus",
    "in the pilot programme", "during the field study", "in the lab",
    "across three factories",
]

# Deterministic fact fragments used to build coherent passages.
FACT_TEMPLATES = [
    "A team of {subject} {verb} the effect of {obj} {place}.",
    "The study found that {obj} increased by {pct}% after the intervention {place}.",
    "Participants reported higher {obj} when the new method was adopted {place}.",
    "Compared with the baseline, the approach raised {obj} by roughly {pct} percentage points.",
    "Earlier work had suggested a smaller gain, but this result shows {obj} can improve markedly.",
    "The authors note that {obj} depends on consistent practice {place}.",
    "Further analysis indicated the change was statistically significant for {obj}.",
    "These findings may help planners improve {obj} in similar settings.",
]

NUMBER_WORDS = [
    "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve",
]


def make_passage(rng: random.Random, topic: str, sentences: int = 5) -> str:
    """Build a coherent, self-contained passage about *topic*."""
    subj = rng.choice(SUBJECTS)
    obj = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    verb = rng.choice(VERBS)
    parts: List[str] = [f"This article is about {topic}."]
    for _ in range(sentences):
        t = rng.choice(FACT_TEMPLATES)
        parts.append(t.format(
            subject=subj, verb=verb, obj=obj, place=place, pct=rng.randint(8, 47),
        ))
    return " ".join(parts)


def make_summary(rng: random.Random, passage: str, topic: str) -> str:
    """Produce a faithful one-to-two sentence summary of *passage*."""
    first = passage.split(". ")[1] if ". " in passage else passage
    lead = first.rstrip(".") + "."
    stat = rng.choice([
        "The work highlights practical steps that improve outcomes in this area.",
        "Overall, the report stresses consistency and measurement.",
        "In short, the approach shows measurable benefits worth wider adoption.",
        "The piece concludes that small, repeated changes compound into real gains.",
    ])
    return f"{lead} {stat}"


# ---------------------------------------------------------------------------
# Translation dictionaries (closed vocabulary -> full coverage)
# ---------------------------------------------------------------------------
# We translate only within a small closed vocabulary so every word is known.
# This yields grammatical, fully-translatable sentences (offline fallback).

EN_WORDS = {
    # determiners / pronouns
    "the": None, "a": None, "an": None, "we": None, "they": None, "he": None,
    "she": None, "it": None,
    # subjects
    "cat": "noun", "dog": "noun", "bird": "noun", "fish": "noun",
    "child": "noun", "student": "noun", "teacher": "noun", "farmer": "noun",
    "engineer": "noun", "book": "noun", "river": "noun", "mountain": "noun",
    "city": "noun", "sun": "noun",
    # verbs (base)
    "eats": "verb", "sees": "verb", "reads": "verb", "writes": "verb",
    "opens": "verb", "closes": "verb", "likes": "verb", "helps": "verb",
    "builds": "verb", "finds": "verb",
    # objects
    "food": "noun", "water": "noun", "letter": "noun", "story": "noun",
    "door": "noun", "window": "noun", "friend": "noun", "book": "noun",
    # adjectives
    "small": "adj", "large": "adj", "red": "adj", "blue": "adj",
    "green": "adj", "old": "adj", "new": "adj", "happy": "adj",
    # connectors
    "and": None, "but": None, "because": None,
}

# Target-language lexicons keyed by English lemma -> surface form.
LEXICONS: Dict[str, Dict[str, str]] = {
    "French": {
        "cat": "chat", "dog": "chien", "bird": "oiseau", "fish": "poisson",
        "child": "enfant", "student": "étudiant", "teacher": "professeur",
        "farmer": "agriculteur", "engineer": "ingénieur", "book": "livre",
        "river": "rivière", "mountain": "montagne", "city": "ville", "sun": "soleil",
        "eats": "mange", "sees": "voit", "reads": "lit", "writes": "écrit",
        "opens": "ouvre", "closes": "ferme", "likes": "aime", "helps": "aide",
        "builds": "construit", "finds": "trouve",
        "food": "nourriture", "water": "eau", "letter": "lettre", "story": "histoire",
        "door": "porte", "window": "fenêtre", "friend": "ami", "book2": "livre",
        "small": "petit", "large": "grand", "red": "rouge", "blue": "bleu",
        "green": "vert", "old": "vieux", "new": "nouveau", "happy": "heureux",
        "and": "et", "but": "mais", "because": "parce que",
        "the": "le", "a": "un", "an": "un", "we": "nous", "they": "ils",
        "he": "il", "she": "elle", "it": "il",
    },
    "German": {
        "cat": "Katze", "dog": "Hund", "bird": "Vogel", "fish": "Fisch",
        "child": "Kind", "student": "Student", "teacher": "Lehrer",
        "farmer": "Bauer", "engineer": "Ingenieur", "book": "Buch",
        "river": "Fluss", "mountain": "Berg", "city": "Stadt", "sun": "Sonne",
        "eats": "isst", "sees": "sieht", "reads": "liest", "writes": "schreibt",
        "opens": "öffnet", "closes": "schließt", "likes": "mag", "helps": "hilft",
        "builds": "baut", "finds": "findet",
        "food": "Essen", "water": "Wasser", "letter": "Brief", "story": "Geschichte",
        "door": "Tür", "window": "Fenster", "friend": "Freund", "book2": "Buch",
        "small": "klein", "large": "groß", "red": "rot", "blue": "blau",
        "green": "grün", "old": "alt", "new": "neu", "happy": "glücklich",
        "and": "und", "but": "aber", "because": "weil",
        "the": "der", "a": "ein", "an": "ein", "we": "wir", "they": "sie",
        "he": "er", "she": "sie", "it": "es",
    },
    "Spanish": {
        "cat": "gato", "dog": "perro", "bird": "pájaro", "fish": "pez",
        "child": "niño", "student": "estudiante", "teacher": "profesor",
        "farmer": "granjero", "engineer": "ingeniero", "book": "libro",
        "river": "río", "mountain": "montaña", "city": "ciudad", "sun": "sol",
        "eats": "come", "sees": "ve", "reads": "lee", "writes": "escribe",
        "opens": "abre", "closes": "cierra", "likes": "gusta", "helps": "ayuda",
        "builds": "construye", "finds": "encuentra",
        "food": "comida", "water": "agua", "letter": "carta", "story": "historia",
        "door": "puerta", "window": "ventana", "friend": "amigo", "book2": "libro",
        "small": "pequeño", "large": "grande", "red": "rojo", "blue": "azul",
        "green": "verde", "old": "viejo", "new": "nuevo", "happy": "feliz",
        "and": "y", "but": "pero", "because": "porque",
        "the": "el", "a": "un", "an": "un", "we": "nosotros", "they": "ellos",
        "he": "él", "she": "ella", "it": "lo",
    },
}

# A few fixed, fully-translated phrase pairs for EN <-> JA / ZH (limited but real).
PHRASE_BANK: List[Dict[str, str]] = [
    {"source": "The cat sleeps.", "target": "猫は寝ています。", "source_lang": "English", "target_lang": "Japanese"},
    {"source": "We read a book.", "target": "私たちは本を読みます。", "source_lang": "English", "target_lang": "Japanese"},
    {"source": "The sun is red.", "target": "太陽は赤いです。", "source_lang": "English", "target_lang": "Japanese"},
    {"source": "The dog eats food.", "target": "犬は食べ物を食べます。", "source_lang": "English", "target_lang": "Japanese"},
    {"source": "The cat sleeps.", "target": "猫在睡觉。", "source_lang": "English", "target_lang": "Chinese"},
    {"source": "We read a book.", "target": "我们读书。", "source_lang": "English", "target_lang": "Chinese"},
    {"source": "The sun is red.", "target": "太阳是红色的。", "source_lang": "English", "target_lang": "Chinese"},
    {"source": "Water is blue.", "target": "水是蓝色的。", "source_lang": "English", "target_lang": "Chinese"},
    {"source": "猫は寝ています。", "target": "The cat sleeps.", "source_lang": "Japanese", "target_lang": "English"},
    {"source": "本を読みます。", "target": "We read a book.", "source_lang": "Japanese", "target_lang": "English"},
    {"source": "太阳是红色的。", "target": "The sun is red.", "source_lang": "Chinese", "target_lang": "English"},
    {"source": "水是蓝色的。", "target": "Water is blue.", "source_lang": "Chinese", "target_lang": "English"},
]

SENTENCE_TEMPLATES = [
    "The {subj} {verb} the {obj}.",
    "A {adj} {subj} {verb} {obj}.",
    "The {subj} {verb} {obj} and the {subj2} {verb2} the {obj2}.",
    "{He} {verb} the {obj} because it is {adj}.",
    "We {verb} a {adj} {obj}.",
]


def available_translation_pairs() -> List[tuple]:
    """Language directions the offline translator can actually handle."""
    pairs = [(f"English", lng) for lng in LEXICONS]
    pairs += [(lng, "English") for lng in LEXICONS]
    return pairs


def translate_en_to(target_lang: str, sentence: str) -> str:
    """Word-level translate of a closed-vocabulary English sentence."""
    lex = LEXICONS.get(target_lang)
    if lex is None:
        return sentence
    out = []
    for w in sentence.split():
        clean = w.strip(".,")
        punct = w[len(clean):]
        tr = lex.get(clean.lower(), clean)
        out.append(tr + punct)
    return " ".join(out)
