# Findings (including negative and engineering results)

## F1 (Phase 3): "no forgetting" is by construction, not a result
Independent modules with frozen parameters cannot interfere. The modular system's 0.000 forgetting drop is therefore uninformative about the hypothesis; the informative comparison is the naive sequential shared-MLP baseline, which collapsed (compare_numbers 1.000 -> 0.038 after learning the next task). A joint-trained shared MLP with fewer parameters (2,932 vs 3,817 total) matched the modular accuracies on all three single tasks. **On non-compositional tasks the modular system shows no accuracy advantage over a conventional tiny network; its advantage is incremental acquisition without retraining and portable packaging.**

## F2 (Phase 4): composition works on the tested tasks, and beats end-to-end baselines where the task is compositional
With human-authored plans, independently trained modules composed reliably: measured composite accuracy matched what independent module errors predict (count_inside 0.957 measured vs 0.968 predicted; mixed 0.987 vs 0.980; A->B->A->C 0.993 vs 0.980), i.e. no sign of representation mismatch or wiring errors. End-to-end MLPs given 3,000 examples were far worse on count_inside (0.57), mixed (0.83), path_ABAC (0.86), equal on sort4 (1.00; the modular version reuses a 1,251-parameter module versus a 6,040-parameter MLP).
Caveats that limit this claim: (a) the **plans are hand-written**; whether a system can *learn or obtain* plans is untested (Phase 5-6 teacher plans, Phase 7 learned routing); (b) modules received per-module labelled supervision from the teacher, the baseline only end-to-end labels, so the comparison measures the value of decomposition plus intermediate supervision, not only architecture; (c) one baseline configuration (MLP 64x64, 3,000 examples, 150 epochs); a stronger baseline (more data/tuning) might close the gap; (d) compare_numbers has a finite 400-pair domain, which all of train/val/test cover, so composites over it test composition rather than generalization of that module.

## F3 (Phase 4): engineering bug found by the overhead benchmark
First measurement: 16-node sort plan overhead 1,443 us vs 156 us of module compute. Cause: `Registry::record_call` rewrote `registry.json` on every inference. Fix: buffer usage statistics in memory and flush on command end/drop (structural changes still persist immediately). After the fix: overhead 30-55 us for the same plan (26x lower). Trade-off: usage statistics can be lost on a crash.

## F4 (Phase 4): latency cost of modularity
Through the same ONNX backend a monolithic end-to-end MLP takes 7-9 us per task; the modular plans take 66-78 us (about 9x), of which module compute is roughly half. Intent routing adds ~10-25 us per plan over id routing. Absolute numbers are small (<150 us), but the modular design is slower than a monolithic network wherever a monolithic network is accurate enough. Energy was not measured.

## F5 (Phase 4): scheduling
Independent branches are identified (`parallel_levels`) but executed sequentially; no parallel speedup is claimed or measured.
