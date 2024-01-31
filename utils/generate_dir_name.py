import json

def generate_dir_name(output_dir,
                    model_list,
                    gen_type, 
                    do_sample, 
                    temperature, 
                    top_p, 
                    top_k,  
                    is_src_statement,
                    max_gen_len):
    
    if '.csv' not in output_dir:
        lang = 'src' if is_src_statement else 'en'
        sample = 'sample' if do_sample else 'greedy'
        model_lst = '_'.join(model_list)
        
        output_dir =  output_dir + gen_type + '_' + model_lst + '_' + lang 
    else:
        output_dir = output_dir[:-4]

    params_dict = {'model_type': gen_type,
                   'model_list': model_list,
                   'do_sample': do_sample,
                   'temperature': temperature,
                   'top_p': top_p,
                   'top_k': top_k,
                   'max_gen_len': max_gen_len,
                   'is_src_statement': is_src_statement}
    
    print(output_dir)
    
    return output_dir, params_dict