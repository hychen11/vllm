# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.


LONG_PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as follows.\n# Table\n" + """

| ID  | Name          | Age | Occupation    | Country       | Email                  | Phone Number   | Address                       |
|-----|---------------|-----|---------------|---------------|------------------------|----------------|------------------------------|
| 1   | John Doe      | 29  | Engineer      | USA           | john.doe@example.com   | 555-1234       | 123 Elm St, Springfield, IL  |
| 2   | Jane Smith    | 34  | Doctor        | Canada        | jane.smith@example.com | 555-5678       | 456 Oak St, Toronto, ON      |
| 3   | Alice Johnson | 27  | Teacher       | UK            | alice.j@example.com    | 555-8765       | 789 Pine St, London, UK      |
| 4   | Bob Brown     | 45  | Artist        | Australia     | bob.b@example.com      | 555-4321       | 321 Maple St, Sydney, NSW    |
| 5   | Carol White   | 31  | Scientist     | New Zealand   | carol.w@example.com    | 555-6789       | 654 Birch St, Wellington, NZ |
| 6   | Dave Green    | 28  | Lawyer        | Ireland       | dave.g@example.com     | 555-3456       | 987 Cedar St, Dublin, IE     |
| 7   | Emma Black    | 40  | Musician      | USA           | emma.b@example.com     | 555-1111       | 246 Ash St, New York, NY     |
| 8   | Frank Blue    | 37  | Chef          | Canada        | frank.b@example.com    | 555-2222       | 135 Spruce St, Vancouver, BC |
| 9   | Grace Yellow  | 50  | Engineer      | UK            | grace.y@example.com    | 555-3333       | 864 Fir St, Manchester, UK   |
| 10  | Henry Violet  | 32  | Artist        | Australia     | henry.v@example.com    | 555-4444       | 753 Willow St, Melbourne, VIC|
| 11  | Irene Orange  | 26  | Scientist     | New Zealand   | irene.o@example.com    | 555-5555       | 912 Poplar St, Auckland, NZ  |
| 12  | Jack Indigo   | 38  | Teacher       | Ireland       | jack.i@example.com     | 555-6666       | 159 Elm St, Cork, IE         |
| 13  | Karen Red     | 41  | Lawyer        | USA           | karen.r@example.com    | 555-7777       | 357 Cedar St, Boston, MA     |
| 14  | Leo Brown     | 30  | Chef          | Canada        | leo.b@example.com      | 555-8888       | 246 Oak St, Calgary, AB      |
| 15  | Mia Green     | 33  | Musician      | UK            | mia.g@example.com      | 555-9999       | 975 Pine St, Edinburgh, UK   |
| 16  | Noah Yellow   | 29  | Doctor        | Australia     | noah.y@example.com     | 555-0000       | 864 Birch St, Brisbane, QLD  |
| 17  | Olivia Blue   | 35  | Engineer      | New Zealand   | olivia.b@example.com   | 555-1212       | 753 Maple St, Hamilton, NZ   |
| 18  | Peter Black   | 42  | Artist        | Ireland       | peter.b@example.com    | 555-3434       | 912 Fir St, Limerick, IE     |
| 19  | Quinn White   | 28  | Scientist     | USA           | quinn.w@example.com    | 555-5656       | 159 Willow St, Seattle, WA   |
| 20  | Rachel Red    | 31  | Teacher       | Canada        | rachel.r@example.com   | 555-7878       | 357 Poplar St, Ottawa, ON    |
| 21  | Steve Green   | 44  | Lawyer        | UK            | steve.g@example.com    | 555-9090       | 753 Elm St, Birmingham, UK   |
| 22  | Tina Blue     | 36  | Musician      | Australia     | tina.b@example.com     | 555-1213       | 864 Cedar St, Perth, WA      |
| 23  | Umar Black    | 39  | Chef          | New Zealand   | umar.b@example.com     | 555-3435       | 975 Spruce St, Christchurch, NZ|
| 24  | Victor Yellow | 43  | Engineer      | Ireland       | victor.y@example.com   | 555-5657       | 246 Willow St, Galway, IE    |
| 25  | Wendy Orange  | 27  | Artist        | USA           | wendy.o@example.com    | 555-7879       | 135 Elm St, Denver, CO       |
| 26  | Xavier Green  | 34  | Scientist     | Canada        | xavier.g@example.com   | 555-9091       | 357 Oak St, Montreal, QC     |
| 27  | Yara Red      | 41  | Teacher       | UK            | yara.r@example.com     | 555-1214       | 975 Pine St, Leeds, UK       |
| 28  | Zack Blue     | 30  | Lawyer        | Australia     | zack.b@example.com     | 555-3436       | 135 Birch St, Adelaide, SA   |
| 29  | Amy White     | 33  | Musician      | New Zealand   | amy.w@example.com      | 555-5658       | 159 Maple St, Wellington, NZ |
| 30  | Ben Black     | 38  | Chef          | Ireland       | ben.b@example.com      | 555-7870       | 246 Fir St, Waterford, IE    |
"""
prompt = """Explain quantum computing concepts using Shakespearean style, covering:
1. Superposition ("To be or not to be" analogies)
2. Entanglement ("Star-crossed qubits")
3. Decoherence ("The slings and arrows of noisy environments")
4. Quantum gates ("The play's the thing, wherein we'll catch the qubits")
5. Algorithms (Like "A Midsummer Night's Dream" but with Grover's search)

Use iambic pentameter where possible and include at least 3 metaphors comparing quantum states to:
- Natural phenomena (e.g. "Qubits dance like leaves in quantum wind")
- Human emotions
- Mythological tales

Structure the response as a soliloquy by a physicist contemplating the nature of reality."""
# Create a sampling params object.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The president of the United States is really like one in the history",
    "The future of AI is",
    
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="/home/hychen11/Desktop/models/llama3-3B",
    # quantization="bitsandbytes",
    dtype="float16",
    kv_cache_dtype="auto",  # 使用fp8减少内存占用
    gpu_memory_utilization=0.98,  # 使用90%的GPU内存
    block_size=16,  # 减小block size以支持更小的显存
    swap_space=4,  # 使用16GB的swap空间
    enforce_eager=True,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {generated_text!r}")
    print("-" * 60)
