"""LoRA fine-tuning of Qwen3-Omni on AT-DIFFER (Methods 1-3 of the paper).

The stage has three steps: ``prepare`` builds a 16 kHz audio cache and the
train / eval manifests, ``train`` trains the LoRA adapter, and ``merge`` merges
the adapter into the base model so that the inference stage can load it as an
ordinary checkpoint.
"""
