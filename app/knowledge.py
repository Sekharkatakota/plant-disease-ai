"""Disease -> solution knowledge base, multilingual."""
import json
from functools import lru_cache

LANGS = {"English": "en", "తెలుగు": "te", "हिन्दी": "hi", "தமிழ்": "ta",
         "ಕನ್ನಡ": "kn", "മലയാളം": "ml", "मराठी": "mr", "বাংলা": "bn"}


class Knowledge:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.db = json.load(f)

    @staticmethod
    @lru_cache(maxsize=2048)
    def _translate(text: str, lang: str) -> str:
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source="en", target=lang).translate(text)
        except Exception:
            return text  # offline -> fall back to English

    def advice(self, label: str, lang: str = "en") -> dict:
        entry = self.db.get(label)
        if entry is None:
            if label.endswith("healthy"):
                entry = self.db["_healthy"]
            elif label.startswith("Background"):
                entry = self.db["_background"]
            else:  # unknown class -> generic advice
                entry = self.db["_generic"]
        if lang in entry:
            return {**entry[lang], "source": "curated"}
        en = entry["en"]  # no curated translation: translate on the fly
        return {
            "summary": self._translate(en["summary"], lang),
            "treatment": [self._translate(t, lang) for t in en["treatment"]],
            "source": "machine-translated",
        }
