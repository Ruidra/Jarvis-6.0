"""
Voice biometrics / speaker verification — REQUIRES SETUP.

Powerful tools (god_mode, computer_control) should only fire for a verified speaker.
This is a scaffold: plug in a speaker-verification model (e.g. a fine-tuned
embedding + cosine threshold, or a cloud service) and call ``verify()`` before
dangerous actions.  Not enabled by default.
"""


class SpeakerVerifier:  # pragma: no cover - scaffold
    def __init__(self, enrolled: dict[str, object] | None = None):
        self.enrolled = enrolled or {}

    def is_enabled(self) -> bool:
        return bool(self.enrolled)

    def enroll(self, user: str, embedding: object) -> None:
        self.enrolled[user] = embedding

    def verify(self, embedding: object) -> str | None:
        if not self.enrolled:
            return None  # disabled -> allow (no verification)
        raise NotImplementedError("Implement cosine/threshold match against enrolled.")
