import json, pathlib

state_path = pathlib.Path('models/bhojpuri-whisper-small-full/checkpoint-14900/trainer_state.json')
with open(state_path) as f:
    state = json.load(f)

log = state['log_history']
evals = [e for e in log if 'eval_wer' in e]
print(f"Total eval checkpoints: {len(evals)}")
print(f"Best WER: {state['best_metric']*100:.2f}%  @ step {state['best_global_step']} (epoch {state['epoch']:.3f})")
print()
print("--- Eval Progression ---")
for e in evals:
    print(f"  Step {int(e['step']):>6} | Epoch {e['epoch']:.3f} | Loss {e['eval_loss']:.4f} | WER {e['eval_wer']*100:.2f}%")

import os
model_path = pathlib.Path('models/bhojpuri-whisper-small-full/checkpoint-14900/model.safetensors')
size_mb = model_path.stat().st_size / (1024*1024)
print(f"\nModel size: {size_mb:.1f} MB ({size_mb/1024:.2f} GB)")

# Count parameters from config
cfg_path = pathlib.Path('models/bhojpuri-whisper-small-full/checkpoint-14900/config.json')
with open(cfg_path) as f:
    cfg = json.load(f)
d = cfg['d_model']
enc_layers = cfg['encoder_layers']
dec_layers = cfg['decoder_layers']
enc_heads = cfg['encoder_attention_heads']
dec_heads = cfg['decoder_attention_heads']
print(f"\nModel Architecture (from config):")
print(f"  d_model (hidden size): {d}")
print(f"  Encoder layers: {enc_layers}  | heads: {enc_heads}")
print(f"  Decoder layers: {dec_layers}  | heads: {dec_heads}")
print(f"  Encoder FFN dim: {cfg['encoder_ffn_dim']}")
print(f"  Decoder FFN dim: {cfg['decoder_ffn_dim']}")
print(f"  Vocab size: {cfg['vocab_size']}")
print(f"  Attention dropout: {cfg['attention_dropout']}")
print(f"  Dropout: {cfg['dropout']}")
