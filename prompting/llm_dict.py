from eval_political_worldviews.prompting.huggingface_llm_wrapper import hf_loader, CausalLLM, Seq2SeqLLM

llm_dict = {'falcon-7b':{'name': 'tiiuae/falcon-7b',
                         'type': CausalLLM,
                        'chat': False

                         },

            'falcon-40b':{'name': 'tiiuae/falcon-40b',
                         'type': CausalLLM,
                        'chat': False
                         },

            'mt5-small':{'name': 'google/mt5-small',
                         'type': Seq2SeqLLM,
                        'chat': False},
                         
            'flan-t5-small': {'name':'google/flan-t5-small',
                              'type': Seq2SeqLLM,
                              'chat': False},

            'llama-2-7b': {'name': 'meta-llama/Llama-2-7b-hf',
                           'type': CausalLLM,
                           'chat': False
                            },

            'llama-2-13b': {'name': 'meta-llama/Llama-2-13b-hf',
                           'type': CausalLLM,
                           'chat': False
                            },

            'llama-2-70b': {'name': 'meta-llama/Llama-2-70b-hf',
                           'type': CausalLLM,
                           'chat': False
                            },

            'llama-2-7b-chat': {'name': 'meta-llama/Llama-2-7b-chat-hf',
                           'type': CausalLLM,
                           'chat': True
                            },

            'llama-2-13b-chat': {'name': 'meta-llama/Llama-2-13b-chat-hf',
                           'type': CausalLLM,
                           'chat': True
                            },

            'llama-2-70b-chat': {'name': 'meta-llama/Llama-2-70b-chat-hf',
                           'type': CausalLLM,
                           'chat': True
                            }
            }