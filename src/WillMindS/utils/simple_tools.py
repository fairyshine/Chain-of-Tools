
def find_all_idx_in_list(input_list,element):
    result_list=[]
    for i in range(len(input_list)):
        if input_list[i] == element:
            result_list.append(i)
    return result_list