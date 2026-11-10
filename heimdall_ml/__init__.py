"""Heimdall classifier tools: data access, the tonal baseline, alignment, mixing and evaluation.

Every module works for any model. The firmware is the reference: `tonal` follows MAX78002/audio_features.c,
including its 48-to-16 kHz decimator, so offline data matches what the board sees.
"""
