# Raspberry Pi Documentation Chatbot

This is a prototype of a documentation bot that can take in natural language queries and find the most relevant help files from the official Raspberry Pi documentation.

## Install

I recommend using a virtual environment, like `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

The `question_search.py` script presents a query prompt. Type a question in and press return to see the results. Here are a couple of examples:

```bash
python question_search.py
```

```bash
Loading question embeddings...
Loaded 7588 questions.
Downloading/loading embedding model...
config.json: 1.49kB [00:00, 1.50MB/s]
Query> What do the LED patterns on boot mean?
Query: What do the LED patterns on boot mean?
  1. [0.2111] The green LED keeps flashing in a pattern when I try to boot. How do I figure out what it's trying to tell me?
      computers/raspberry-pi/boot-eeprom.faq#diagnostics
  2. [0.2437] My Pi is flashing an LED in a pattern instead of booting. What does that mean?
      computers/configuration.faq#boot-folder-contents
  3. [0.2566] My Pi shows an error pattern on the LED and won't boot. What does that mean?
      computers/raspberry-pi/bootflow-eeprom.faq#second-stage-bootloader

Query> How do I send audio through the headphone port?
Query: How do I send audio through the headphone port?
  1. [0.2183] How do I force audio to come out of the headphone jack instead of HDMI?
      computers/os.faq#specify-an-audio-output-device
  2. [0.2343] My audio is coming out through HDMI but I want it through the headphone jack. How do I change that?
      computers/os/playing-audio-and-video.faq#vlc-gui
  3. [0.2347] My audio is coming out of HDMI but I want it through the headphone jack. How do I switch that?
      computers/configuration/audio-config.faq#audio
```

## Implementation

In the future I'll hook up Moonshine Voice to provide a speech interface, but I wanted to ensure we had a good retrieval foundation first.

I'm using a "hypothetical questions" approach to retrieval, where I use an LLM to generate potential questions that are answered by each documentation section. I then use sentence embeddings to find the closest hypothetical questions to the actual user queries, and display them along with the documentation section they were generated from.

The distance between the user and documentation questions is shown by the score prefix, and the documentation link is in the suffix.
