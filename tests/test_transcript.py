from fakes import final, interim

from earpiece.brain.transcript import TranscriptStore


def make_store() -> TranscriptStore:
    return TranscriptStore(mission="help me with trivia")


def test_interim_updates_ui_only():
    store = make_store()
    assert store.add(interim("THEM", "what year was")) is None
    assert store.interim["THEM"] == "what year was"
    assert store.utterances == []


def test_final_clears_interim_and_records_utterance():
    store = make_store()
    store.add(interim("THEM", "what year"))
    utt = store.add(final("THEM", "What year was the transistor invented?"))
    assert utt is not None
    assert "THEM" not in store.interim
    assert store.utterances[-1].text == "What year was the transistor invented?"


def test_as_messages_is_append_only():
    store = make_store()
    store.add(final("THEM", "Hello there."))
    first = store.as_messages()
    assert first[0]["role"] == "system"
    assert "help me with trivia" in first[0]["content"]
    assert first[1]["role"] == "user"
    assert "THEM: Hello there." in first[1]["content"]

    store.add_answer("Hi.", interrupted=False)
    store.add(final("ME", "Good morning."))
    second = store.as_messages()

    # Prefix is byte-stable: earlier messages unchanged, new content appended.
    assert second[: len(first)][0] == first[0]
    assert second[1] == first[1]
    assert second[2] == {"role": "assistant", "content": "Hi."}
    assert "ME: Good morning." in second[3]["content"]


def test_interrupted_answer_gets_marker():
    store = make_store()
    store.add(final("THEM", "Question?"))
    store.as_messages()
    store.add_answer("Partial answ", interrupted=True)
    messages = store.as_messages()
    assert messages[-1]["content"].endswith("[interrupted — conversation moved on]")


def test_speaker_tagging_in_lines():
    store = make_store()
    store.add(final("ME", "I think it was 1950."))
    store.add(final("THEM", "Close!"))
    lines = store.recent_lines()
    assert "ME: I think it was 1950." in lines
    assert "THEM: Close!" in lines
