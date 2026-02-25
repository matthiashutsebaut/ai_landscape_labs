"""
Voice Memo Generator for Lab 5
================================
Uses the OpenAI TTS API to generate three realistic voice memos from loan
applicants.  Run this script once before starting Lab 5.

Requirements:
  pip install openai
  export OPENAI_API_KEY=sk-...

Output:
  voice_memos/memo_001_hardship.mp3    — borderline applicant, medical hardship
  voice_memos/memo_002_confident.mp3   — strong applicant, confirms employment
  voice_memos/memo_003_weak.mp3        — weak applicant, adds no new information
"""

from pathlib import Path
from openai import OpenAI

client = OpenAI()

MEMOS = [
    {
        "filename": "memo_001_hardship.mp3",
        "voice": "alloy",
        "text": (
            "Hi, my name is Sarah and I am applying for this loan. "
            "I wanted to give some context about my credit history. "
            "Last year I was diagnosed with a serious illness and had to take "
            "three months off work for treatment and recovery. "
            "During that time I fell behind on a couple of payments. "
            "I am fully recovered now, back to my full-time position, "
            "and my salary is stable. "
            "The loan is to help me clear the medical bills from that period. "
            "I have a letter from my employer confirming my current employment "
            "and salary if that would help. "
            "Thank you for your consideration."
        ),
    },
    {
        "filename": "memo_002_confident.mp3",
        "voice": "echo",
        "text": (
            "Good morning. I am applying for a car loan. "
            "I have been with my current employer for eight years now "
            "as a senior software engineer. "
            "My salary has been growing every year and I currently earn "
            "well above the loan amount. "
            "I have no outstanding debts and I pay all my bills on time. "
            "The car is for my daily commute to the office which is about "
            "forty kilometres each way. "
            "I have attached all required documents. "
            "I am confident this is a straightforward application."
        ),
    },
    {
        "filename": "memo_003_weak.mp3",
        "voice": "fable",
        "text": (
            "Uh, hi. I need this loan. "
            "I have had some problems in the past with money but I want to try again. "
            "I do not really have much more to say. "
            "I hope you will approve it. "
            "Thank you."
        ),
    },
]

output_dir = Path(__file__).parent / "voice_memos"
output_dir.mkdir(exist_ok=True)

for memo in MEMOS:
    out_path = output_dir / memo["filename"]
    if out_path.exists():
        print(f"  Skipping (already exists): {out_path.name}")
        continue

    print(f"  Generating: {out_path.name} ...")
    response = client.audio.speech.create(
        model="tts-1",
        voice=memo["voice"],
        input=memo["text"],
    )
    response.stream_to_file(out_path)
    print(f"  Saved: {out_path}")

print("\nDone. Voice memos are ready in:", output_dir)
