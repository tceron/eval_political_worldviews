from openai import OpenAI
import tiktoken
from tabulate import tabulate

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def total_num_tokens_from_messages(messages, 
                                   model="gpt-3.5-turbo-0613", 
                                   num_return_sequences=30, 
                                   max_gen_length=20,
                                   input_token_rate=0.001,
                                   output_token_rate=0.002):
    
    # Caluculate input token num
    input_token_num = 0

    for message in messages:
        input_token_num += num_tokens_from_messages(message, model)
    
    total_input_token_num = num_return_sequences * input_token_num

    # Calculate output token num
    total_output_token_num = len(messages) * max_gen_length * num_return_sequences

    # Calculate total token num
    total_token_num = total_input_token_num + total_output_token_num

    input_token_cost = (total_input_token_num / 1000) * input_token_rate
    output_token_cost = (total_output_token_num / 1000) * output_token_rate

    total_cost = input_token_cost + output_token_cost

    cost_table = [['input tokens', total_input_token_num, input_token_cost],
                  ['output tokens', total_output_token_num, output_token_cost],
                  ['total tokens', total_token_num, total_cost]]
    
    print(tabulate(cost_table, 
                   headers=["Token type", "Total token count", "Total token cost ($)"],
                   tablefmt='grid'))
    

    
    return total_cost