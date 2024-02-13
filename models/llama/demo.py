from transformers import AutoTokenizer, LlamaForCausalLM

if __name__ == '__main__':

    model = LlamaForCausalLM.from_pretrained('/projekte/tcl/users/baricaa/mardy/actor_id/llm/llama/weights/')
    tokenizer = AutoTokenizer.from_pretrained('/projekte/tcl/users/baricaa/mardy/actor_id/llm/llama/weights/')

    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=100)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])