# high-level stuff i noticed:
- we need to import the logging granularity introduced in main to latency-bound inference.
- extend to containerised and mult-tenant scenarios
- use the clean loop from deadline-harness and integrate the phys ai angle to the same.
- extend profiling: ncu (kernel level) and nsys (system level).
- measure the overhead of our CPU side (asyncio); explore ZMQ if it is significant.
- flame graphs in logging: within one session/request's lifetime, show where each process spends its time.
- more verbose logging for deadline misses: why did it miss, where did the model go off-course?
- power matters for both the space and physical AI angles.
- the wakeup of the feeder can be an issue! how can we make sure it happens more deterministically?