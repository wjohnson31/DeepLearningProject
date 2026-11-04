# Measured dense baseline evidence

D1 has 2,590,848 unique trainable parameters. It was trained from scratch on 45,852 TinyStories stories (10,000,138 byte-level BPE tokens), using a frozen 4096-entry vocabulary. Training completed 5,000 updates with 20,480,000 target-token exposures. The lowest development CE was 2.437512 nats (perplexity 11.444537) at update 5,000. Development evaluation covered 200,064 target tokens. Reloading the selected checkpoint reproduced CE with absolute difference 0.

The recorded environment was cpu, fp32, with four PyTorch CPU threads. Measured training time across the pilot and resumed run was 33.45 minutes. Throughput in the resumed session, excluding its first 20 updates, was 10,028.3 target tokens/second. The analytical forward matrix-operation estimate is 5,505,024 FLOPs/token, counting two operations per multiply-add and excluding scalar nonlinearities and normalization. GPU throughput and peak GPU memory were not measured.

These are single-seed dense-baseline results. They do not establish an MoE improvement, a balancing-schedule benefit, or statistical robustness. The final-test partition has not been evaluated. A paired MoE comparison must preserve this protocol or rerun the dense control with the revised one.

Raw evidence: baseline_results.json, baseline_evaluations.json, verification.json, dataset_summary.json, and figures/learning_curve.png. All reported metrics originate from the saved checkpoints and logs.
