# Background: concepts behind the metrics

## Prefill and decode
Generating a reply has two phases. **Prefill** processes the whole prompt in parallel and determines the time to first token (TTFT). **Decode** generates one token at a time and determines tokens per second after the first token. Long prompts make prefill expensive; long answers make decode expensive.

## Continuous batching
A serving engine can add new requests to a running batch at every decoding step instead of waiting for the batch to finish. This keeps the GPU busy and is why aggregate throughput usually rises as concurrency increases, while each individual request may get slower.

## KV cache and PagedAttention
During generation the model stores key and value tensors for every token it has seen (the KV cache). Its size grows with sequence length and with the number of concurrent requests, and it is often the limit on how many requests fit in GPU memory. PagedAttention stores the cache in fixed-size blocks to reduce fragmentation and wasted memory.

## Prefix caching
When many requests start with the same text, such as a shared system prompt, the engine can reuse the cached computation for that prefix. This reduces prefill work and should lower TTFT. The benchmark's shared system prompt exists to make this effect measurable.

## Quantization
Quantization stores model weights with fewer bits. 4-bit methods such as AWQ use roughly a quarter of the memory of fp16 weights and can increase decode speed because less data moves from memory. The cost is possible accuracy loss, which is why this project measures quality alongside speed.

## Latency versus throughput
Raising concurrency increases total tokens per second until the GPU is saturated, after which per-request latency grows without throughput gains. The right operating point depends on the latency target of the application.

## Reading the confidence intervals
Quality is measured on a finite sample. A bootstrap confidence interval shows how much the exact-match rate could vary by chance. If two configurations' intervals overlap heavily, the data do not support a claim that one is more accurate.
