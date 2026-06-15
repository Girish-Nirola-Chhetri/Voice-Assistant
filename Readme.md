# Voice Command Assistant

A 50M parameter encoder-decoder transformer built from scratch that converts natural language voice and text commands into structured JSON outputs.

## What It Does

User gives a command → model reads it → outputs structured JSON → action executor performs the task.

Example:
- Input: "set alarm for 5am tomorrow"
- Output: `{"intent": "set_alarm", "parameters": {"time": "05:00", "date": "tomorrow"}}`

## Current Scope

- Alarm intent category only
- Full pipeline: dataset → tokenizer → model → training → evaluation
- No pretrained weights — everything built from scratch

## Stack

- Python
- PyTorch
- Custom BPE Tokenizer
