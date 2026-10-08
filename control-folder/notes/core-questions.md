# Core questions (v2, 2026-10-06)

1. **Where does it fail?** When inference frameworks are given a latency budget,
   where do they miss it: the model, stale inputs, co-located load, transport,
   or the device's energy and thermal limits?

2. **Why is it variable?** What makes inference latency variable on a shared
   accelerator, and how much of it comes from the model versus everything
   sharing the chip with it?

3. **What fixes it?** How much of the FPGA's determinism can the GPU be given
   directly, and for what remains, which parts of inference should move to an
   FPGA, at what latency and energy cost?

Arc: Q1 locates the misses, Q2 attributes them, Q3 removes them, cheaper step
(determinism on the GPU) before the expensive one (split to the FPGA).
