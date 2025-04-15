# Create an LLM.
llm = LLM(
    model=args.model,
    tokenizer=args.tokenizer,
    trust_remote_code=args.trust_remote_code,
    download_dir=args.download_dir,
    use_np_weights=args.use_np_weights,
    use_dummy_weights=args.use_dummy_weights,
    dtype=args.dtype,
    quantization="bitsandbytes",  # 添加bitsandbytes量化
    kv_cache_dtype="fp8",  # 使用fp8 KV缓存
    seed=args.seed,
    gpu_memory_utilization=args.gpu_memory_utilization,
    swap_space=args.swap_space,
    enforce_eager=args.enforce_eager,
    max_context_len_to_capture=args.max_context_len_to_capture,
    max_num_batched_tokens=args.max_num_batched_tokens,
    max_num_seqs=args.max_num_seqs,
    disable_log_stats=not args.enable_log_stats,
    revision=args.revision,
    code_revision=args.code_revision,
    tokenizer_revision=args.tokenizer_revision,
    max_model_len=args.max_model_len,
    block_size=args.block_size,
    enable_prefix_caching=args.enable_prefix_caching,
    enable_chunked_prefill=args.enable_chunked_prefill,
    disable_sliding_window=args.disable_sliding_window,
    enable_cuda_graph=args.enable_cuda_graph,
    disable_custom_all_reduce=args.disable_custom_all_reduce,
    tensor_parallel_size=args.tensor_parallel_size,
    pipeline_parallel_size=args.pipeline_parallel_size,
    worker_use_ray=args.worker_use_ray,
    max_parallel_loading_workers=args.max_parallel_loading_workers,
    disable_log_requests=not args.enable_log_requests,
    guided_decoding_backend=args.guided_decoding_backend,
    lora_modules=args.lora_modules,
    max_lora_rank=args.max_lora_rank,
    max_cpu_loras=args.max_cpu_loras,
    max_loras=args.max_loras,
    max_num_seqs=args.max_num_seqs,
    max_paddings=args.max_paddings,
    device=args.device,
    ray_workers_use_nsight=args.ray_workers_use_nsight,
    disable_custom_all_reduce=args.disable_custom_all_reduce,
    served_model_name=args.served_model_name,
    max_num_batched_tokens=args.max_num_batched_tokens,
    max_num_seqs=args.max_num_seqs,
    max_paddings=args.max_paddings,
    speculative_config=speculative_config,
    num_scheduler_steps=args.num_scheduler_steps,
    multi_step_stream_outputs=args.multi_step_stream_outputs,
    enable_prefix_caching=args.enable_prefix_caching,
    chunked_prefill_enabled=args.enable_chunked_prefill,
    use_async_output_proc=args.use_async_output_proc,
    disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    mm_processor_kwargs=mm_processor_kwargs,
    pooler_config=pooler_config,
    compilation_config=compilation_config,
    use_cached_outputs=args.use_cached_outputs,
) 

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM benchmark server.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        help="trust remote code from huggingface")
    parser.add_argument("--download-dir",
                        type=str,
                        default=None,
                        help="directory to download and load the weights")
    parser.add_argument("--use-np-weights",
                        action="store_true",
                        help="save a numpy copy of model weights")
    parser.add_argument("--use-dummy-weights",
                        action="store_true",
                        help="use dummy values for model weights")
    parser.add_argument("--dtype",
                        type=str,
                        default="auto",
                        choices=[
                            "auto", "half", "float16", "bfloat16", "float",
                            "float32"
                        ])
    parser.add_argument("--quantization",
                        type=str,
                        default="bitsandbytes",
                        choices=["bitsandbytes", "awq", "gptq", "squeezellm", "fp8"],
                        help="quantization method to use")
    parser.add_argument("--kv-cache-dtype",
                        type=str,
                        default="fp8",
                        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp16", "float32"],
                        help="data type for kv cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.98,
                        help="the percentage of GPU memory to be used for the model")
    parser.add_argument("--swap-space",
                        type=int,
                        default=4,
                        help="CPU swap space size (GiB) per GPU")
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager mode and disable CUDA graph")
    parser.add_argument("--max-context-len-to-capture",
                        type=int,
                        default=8192,
                        help="maximum context length covered by CUDA graphs")
    parser.add_argument("--max-num-batched-tokens",
                        type=int,
                        default=2560,
                        help="maximum number of batched tokens per iteration")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=256,
                        help="maximum number of sequences per iteration")
    parser.add_argument("--disable-log-stats",
                        action="store_true",
                        help="disable logging statistics")
    parser.add_argument("--revision",
                        type=str,
                        default=None,
                        help="model revision")
    parser.add_argument("--code-revision",
                        type=str,
                        default=None,
                        help="code revision")
    parser.add_argument("--tokenizer-revision",
                        type=str,
                        default=None,
                        help="tokenizer revision")
    parser.add_argument("--max-model-len",
                        type=int,
                        default=None,
                        help="model context length")
    parser.add_argument("--block-size",
                        type=int,
                        default=16,
                        help="token block size")
    parser.add_argument("--enable-prefix-caching",
                        action="store_true",
                        help="enable prefix caching")
    parser.add_argument("--enable-chunked-prefill",
                        action="store_true",
                        help="enable chunked prefill")
    parser.add_argument("--disable-sliding-window",
                        action="store_true",
                        help="disable sliding window")
    parser.add_argument("--enable-cuda-graph",
                        action="store_true",
                        help="enable CUDA graph")
    parser.add_argument("--disable-custom-all-reduce",
                        action="store_true",
                        help="disable custom all reduce")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=1,
                        help="tensor parallel size")
    parser.add_argument("--pipeline-parallel-size",
                        type=int,
                        default=1,
                        help="pipeline parallel size")
    parser.add_argument("--worker-use-ray",
                        action="store_true",
                        help="use Ray for distributed serving")
    parser.add_argument("--max-parallel-loading-workers",
                        type=int,
                        default=None,
                        help="maximum number of workers to load model")
    parser.add_argument("--disable-log-requests",
                        action="store_true",
                        help="disable logging requests")
    parser.add_argument("--guided-decoding-backend",
                        type=str,
                        default="outlines",
                        choices=["outlines", "lm-format-enforcer"],
                        help="backend for guided decoding")
    parser.add_argument("--lora-modules",
                        type=str,
                        default=None,
                        help="LoRA module configurations")
    parser.add_argument("--max-lora-rank",
                        type=int,
                        default=16,
                        help="maximum LoRA rank")
    parser.add_argument("--max-cpu-loras",
                        type=int,
                        default=2,
                        help="maximum number of LoRAs to store in CPU memory")
    parser.add_argument("--max-loras",
                        type=int,
                        default=1,
                        help="maximum number of LoRAs to store in GPU memory")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=256,
                        help="maximum number of sequences per iteration")
    parser.add_argument("--max-paddings",
                        type=int,
                        default=64,
                        help="maximum number of paddings in a batch")
    parser.add_argument("--device",
                        type=str,
                        default="auto",
                        choices=["auto", "cuda", "cpu", "neuron"],
                        help="device type")
    parser.add_argument("--ray-workers-use-nsight",
                        action="store_true",
                        help="use nsight for Ray workers")
    parser.add_argument("--disable-custom-all-reduce",
                        action="store_true",
                        help="disable custom all reduce")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="model name used in the API")
    parser.add_argument("--max-num-batched-tokens",
                        type=int,
                        default=2560,
                        help="maximum number of batched tokens per iteration")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=256,
                        help="maximum number of sequences per iteration")
    parser.add_argument("--max-paddings",
                        type=int,
                        default=64,
                        help="maximum number of paddings in a batch")
    parser.add_argument("--speculative-config",
                        type=str,
                        default=None,
                        help="speculative decoding configuration")
    parser.add_argument("--num-scheduler-steps",
                        type=int,
                        default=1,
                        help="number of scheduler steps")
    parser.add_argument("--multi-step-stream-outputs",
                        action="store_true",
                        help="enable multi-step stream outputs")
    parser.add_argument("--enable-prefix-caching",
                        action="store_true",
                        help="enable prefix caching")
    parser.add_argument("--enable-chunked-prefill",
                        action="store_true",
                        help="enable chunked prefill")
    parser.add_argument("--use-async-output-proc",
                        action="store_true",
                        help="use async output processor")
    parser.add_argument("--disable-mm-preprocessor-cache",
                        action="store_true",
                        help="disable multimodal preprocessor cache")
    parser.add_argument("--mm-processor-kwargs",
                        type=str,
                        default=None,
                        help="multimodal processor arguments")
    parser.add_argument("--pooler-config",
                        type=str,
                        default=None,
                        help="pooler configuration")
    parser.add_argument("--compilation-config",
                        type=str,
                        default=None,
                        help="compilation configuration")
    parser.add_argument("--use-cached-outputs",
                        action="store_true",
                        help="use cached outputs")
    return parser 