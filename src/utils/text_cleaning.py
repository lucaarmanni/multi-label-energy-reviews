import re
import emoji

def clean_text(text: str) -> str:
    """
    Cleans review text while preserving casing and punctuation.
    - Removes URLs, emojis, and unwanted characters.
    - Normalizes multiple spaces.
    """
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = emoji.replace_emoji(text, "")
    text = re.sub(r"[^a-zA-ZàèéìòùÀÈÉÌÒÙ0-9\s\.,!?;:'\"()-]", "", text)
    return re.sub(r"\s+", " ", text).strip()
