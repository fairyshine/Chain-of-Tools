
from rich.progress import track

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from WillMindS.utils.io import read_JSON

from corpus import judge_Dataset, judge_Collater, \
                    retriever_Dataset, retriever_Collater

def tool_judge_train(config, logger, model, dataset_dir_dict, mode="train+test"):
    logger.info("============ Training tool judger ============")

    optimizer = optim.AdamW(model.tool_judge.parameters(), lr=config.lr)
    collater = judge_Collater(model.tokenizer)

    if mode != "test":
        raw_dataset = []
        for dataset_name in dataset_dir_dict:
            raw_dataset.extend(read_JSON(dataset_dir_dict[dataset_name]+"train.jsonl"))
        train_dataset = judge_Dataset(raw_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collater)

    for epoch in range(1, config.train_epoch+1):
        if mode !="test":
            model.train()
            counter_dict = {"correct": 0,
                            "predict": 0,
                            "gold": 0}
            logger.info("------ Training Epoch {} ------".format(epoch))
            for step, data in track(enumerate(train_dataloader),description='Training epoch {} ...'.format(epoch)):
                all_steps = len(train_dataloader)
                for key,_ in data.items():
                    data[key] = data[key].cuda()
                with torch.no_grad():
                        foundation_output = model.foundation_model(data["input_ids"], output_hidden_states=True)
                judge_logits = model.tool_judging(foundation_output.hidden_states[-1][0])
                judge_logits = torch.sigmoid(judge_logits)
                loss = F.binary_cross_entropy(judge_logits, data["judge_labels"][0].float())
                loss.backward()

                if step % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.tool_judge.parameters(), config.max_grad_norm)
                    optimizer.step()
                    model.zero_grad()

                correct_num, predict_num, gold_num = tool_judge_eval_calculate(judge_logits.round().int(), data["judge_labels"][0])
                counter_dict["correct"] += correct_num
                counter_dict["predict"] += predict_num
                counter_dict["gold"] += gold_num

                if step %100 == 0 and step > 1:
                    # p = counter_dict["correct"] / (counter_dict["predict"] + 1e-10)
                    # r = counter_dict["correct"] / (counter_dict["gold"] + 1e-10)
                    f1 = 2 * counter_dict["correct"] / (counter_dict["predict"] + counter_dict["gold"] + 1e-10)
                    logger.info('step:{}/{}   '.format(step+1,all_steps)+'loss:'+str(loss.item())+' F1:'+str(f1))
                    counter_dict["correct"], counter_dict["predict"], counter_dict["gold"] = 0, 0, 0

        if mode != "train":
            for dataset_name in dataset_dir_dict:
                dev_dataset = judge_Dataset(read_JSON(dataset_dir_dict[dataset_name]+'dev.jsonl'))
                eval_counter_dict = {"correct": 0,
                                    "predict": 0,
                                    "gold": 0}
                model.eval()
                dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True, collate_fn=collater)
                for step, data in track(enumerate(dev_dataloader), description='Evaling ...'):
                    for key,_ in data.items():
                        data[key] = data[key].cuda()
                    with torch.no_grad():
                        foundation_output = model.foundation_model(data["input_ids"], output_hidden_states=True)
                        judge_logits = model.tool_judging(foundation_output.hidden_states[-1][0])
                        judge_logits = torch.sigmoid(judge_logits)
                    correct_num, predict_num, gold_num = tool_judge_eval_calculate(judge_logits.round().int(), data["judge_labels"][0])
                    eval_counter_dict["correct"] += correct_num
                    eval_counter_dict["predict"] += predict_num
                    eval_counter_dict["gold"] += gold_num     
                p = eval_counter_dict["correct"] / (eval_counter_dict["predict"] + 1e-10)
                r = eval_counter_dict["correct"] / (eval_counter_dict["gold"] + 1e-10)
                f1 = 2 * eval_counter_dict["correct"] / (eval_counter_dict["predict"] + eval_counter_dict["gold"] + 1e-10)
                logger.info('Eval: P: {}  R: {}  F1: {} '.format(str(p),str(r),str(f1)))
            if mode == "test":
                break
        if epoch>=1:
            model.sl_judge(config.model_dir, "save", "epoch_"+str(epoch))
    

def tool_judge_eval_calculate(predict_labels, gold_labels):
    correct_num = (predict_labels * gold_labels).eq(1).sum()
    predict_num = predict_labels.eq(1).sum()
    gold_num = gold_labels.eq(1).sum()
    return correct_num, predict_num, gold_num


def tool_retriever_train(config, logger, model, dataset_dir_dict, mode="train+test"):
    logger.info("============ Training tool retriever ============")

    if config.tensor_weighting:
        optimizer = optim.AdamW([{'params': model.retriever_query.parameters()}, 
                                {'params': model.retriever_tool_selection.parameters()},
                                # {'params': model.retriever_tool_parameter.parameters()},
                                {'params': model.w_tensor.parameters(), 'lr':config.lr_w_tensor}], lr=config.lr)
    else:
        optimizer = optim.AdamW([{'params': model.retriever_query.parameters()}, 
                                {'params': model.retriever_tool_selection.parameters()},
                                # {'params': model.retriever_tool_parameter.parameters()}
                                ], lr=config.lr)
    collater = retriever_Collater(model.tokenizer, model.tool_database)

    if mode != "test":
        raw_dataset = []
        for dataset_name in dataset_dir_dict:
            raw_dataset.extend(read_JSON(dataset_dir_dict[dataset_name]+"train.jsonl"))
        train_dataset = retriever_Dataset(raw_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch, shuffle=True, collate_fn=collater)

    for epoch in range(1, config.train_epoch+1):
        if mode !="test":
            model.train()
            counter_dict = {"correct": 0,
                            "all": 0}
            log_loss = 0
            logger.info("------ Training Epoch {} ------".format(epoch))
            for step, data in track(enumerate(train_dataloader),description='Training epoch {} ...'.format(epoch)):
                all_steps = len(train_dataloader)
                data["query_tokens"] = data["query_tokens"].to('cuda')
                data["tool_tokens"] = data["tool_tokens"].to('cuda')
                with torch.no_grad():
                    query_output = model.foundation_model(input_ids=data["query_tokens"].input_ids, attention_mask=data["query_tokens"].attention_mask, output_hidden_states=True)
                    tool_output = model.foundation_model(input_ids=data["tool_tokens"].input_ids, attention_mask=data["tool_tokens"].attention_mask, output_hidden_states=True)
                query_pos_ids = [torch.where(data["query_tokens"].input_ids[idx] == model.tokenizer.eos_token_id)[0][0].item() for idx in range(data["query_tokens"].input_ids.size()[0])]
                query_vectors_in = torch.stack([query_output.hidden_states[-1][idx][query_pos_ids[idx]] for idx in range(len(query_pos_ids))], 0).cuda()
                tool_pos_ids = [torch.where(data["tool_tokens"].input_ids[idx] == model.tokenizer.eos_token_id)[0][0].item() for idx in range(data["tool_tokens"].input_ids.size()[0])]
                tool_vectors_in = torch.stack([tool_output.hidden_states[-1][idx][tool_pos_ids[idx]] for idx in range(len(tool_pos_ids))], 0).cuda()

                if config.cal_seq:
                    calculated_query_states = model.retriever_query(query_output.hidden_states[-1].to(next(model.retriever_query.parameters()).device))
                    calculated_tool_states = model.retriever_tool_selection(tool_output.hidden_states[-1].to(next(model.retriever_tool_selection.parameters()).device))
                    calculated_query = torch.stack([calculated_query_states[idx][query_pos_ids[idx]] for idx in range(len(query_pos_ids))], 0).cuda()
                    calculated_tool = torch.stack([calculated_tool_states[idx][tool_pos_ids[idx]] for idx in range(len(tool_pos_ids))], 0).cuda()
                    query_vectors_out = model.calculate_query_vector(query_vectors_in, calculated_tensor=calculated_query)
                    tool_vectors_out = model.calculate_tool_vector(tool_vectors_in, calculated_tensor=calculated_tool)
                else:
                    query_vectors_out = model.calculate_query_vector(query_vectors_in)
                    tool_vectors_out = model.calculate_tool_vector(tool_vectors_in)

                query_vectors_out = torch.nn.functional.normalize(query_vectors_out,p=2,dim=-1)
                tool_vectors_out = torch.nn.functional.normalize(tool_vectors_out,p=2,dim=-1)

                query_vectors_out = query_vectors_out.to(tool_vectors_out.device)
                if not config.similarity_norm:
                    # 原算法
                    scores = torch.matmul(query_vectors_out, torch.transpose(tool_vectors_out, 0, 1))
                else:
                # 考虑模长
                    q_norm_list = [torch.norm(query_vecotr) for query_vecotr in query_vectors_out]
                    t_norm_list = [torch.norm(tool_vector) for tool_vector in tool_vectors_out]
                    norm_matrix = torch.tensor([[((q_norm+t_norm)/2)**2 for t_norm in t_norm_list] for q_norm in q_norm_list]).to(tool_vectors_out.device)
                    scores = torch.matmul(query_vectors_out, torch.transpose(tool_vectors_out, 0, 1)).div(norm_matrix)

                softmax_scores = F.log_softmax(scores, dim=1)

                loss = F.nll_loss(softmax_scores.to(data["gold_labels"].device), data["gold_labels"], reduction="mean")
                loss.backward()

                if step % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.retriever_query.parameters(), config.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model.retriever_tool_selection.parameters(), config.max_grad_norm)
                    if config.tensor_weighting:
                        torch.nn.utils.clip_grad_norm_(model.w_tensor.parameters(), config.max_grad_norm)
                    optimizer.step()
                    model.zero_grad()

                correct_num, all_num = tool_retriever_eval_calculate(softmax_scores.to("cpu"), data["gold_labels"].to("cpu"))
                counter_dict["correct"] += correct_num
                counter_dict["all"] += all_num
                log_loss += loss.item()

                if step %100 == 0 and step > 1:
                    acc = counter_dict["correct"] / (counter_dict["all"] + 1e-10)
                    logger.info('step:{}/{}   '.format(step+1,all_steps)+'loss:'+str(log_loss/100)+' ACC:'+str(acc))
                    counter_dict["correct"], counter_dict["all"] = 0, 0
                    log_loss = 0

        if mode != "train":
            for dataset_name in dataset_dir_dict:
                dev_dataset = retriever_Dataset(read_JSON(dataset_dir_dict[dataset_name]+'dev.jsonl'))
                eval_counter_dict = {"correct": 0, "all": 0}
                eval_counter_dict_alltools = {"correct": 0, "correct_top_k": 0, "all": 0}
                model.eval()
                model.calculate_database([dataset_name])
                if config.tensor_weighting and config.tensor_filtering:
                    model.get_tensor_filtering_index()
                dev_dataloader = DataLoader(dev_dataset, batch_size=config.train_batch, shuffle=True, collate_fn=collater)
                for step, data in track(enumerate(dev_dataloader), description='Evaling {}...'.format(dataset_name)):
                    data["query_tokens"] = data["query_tokens"].to('cuda')
                    data["tool_tokens"] = data["tool_tokens"].to('cuda')
                    with torch.no_grad():
                        query_output = model.foundation_model(input_ids=data["query_tokens"].input_ids, attention_mask=data["query_tokens"].attention_mask, output_hidden_states=True)
                        tool_output = model.foundation_model(input_ids=data["tool_tokens"].input_ids, attention_mask=data["tool_tokens"].attention_mask, output_hidden_states=True)
                    query_pos_ids = [torch.where(data["query_tokens"].input_ids[idx] == model.tokenizer.eos_token_id)[0][0].item() for idx in range(data["query_tokens"].input_ids.size()[0])]
                    query_vectors_in = torch.stack([query_output.hidden_states[-1][idx][query_pos_ids[idx]] for idx in range(len(query_pos_ids))], 0).cuda()
                    tool_pos_ids = [torch.where(data["tool_tokens"].input_ids[idx] == model.tokenizer.eos_token_id)[0][0].item() for idx in range(data["tool_tokens"].input_ids.size()[0])]
                    tool_vectors_in = torch.stack([tool_output.hidden_states[-1][idx][tool_pos_ids[idx]] for idx in range(len(tool_pos_ids))], 0).cuda()

                    if config.cal_seq:
                        calculated_query_states = model.retriever_query(query_output.hidden_states[-1].to(next(model.retriever_query.parameters()).device))
                        calculated_tool_states = model.retriever_tool_selection(tool_output.hidden_states[-1].to(next(model.retriever_tool_selection.parameters()).device))
                        calculated_query = torch.stack([calculated_query_states[idx][query_pos_ids[idx]] for idx in range(len(query_pos_ids))], 0).cuda()
                        calculated_tool = torch.stack([calculated_tool_states[idx][tool_pos_ids[idx]] for idx in range(len(tool_pos_ids))], 0).cuda()
                        query_vectors_out = model.calculate_query_vector(query_vectors_in, calculated_tensor=calculated_query)
                        tool_vectors_out = model.calculate_tool_vector(tool_vectors_in, calculated_tensor=calculated_tool)
                    else:
                        query_vectors_out = model.calculate_query_vector(query_vectors_in)
                        tool_vectors_out = model.calculate_tool_vector(tool_vectors_in)

                    query_vectors_out = torch.nn.functional.normalize(query_vectors_out,p=2,dim=-1)
                    tool_vectors_out = torch.nn.functional.normalize(tool_vectors_out,p=2,dim=-1)

                    query_vectors_out = query_vectors_out.to(tool_vectors_out.device)
                    if not config.similarity_norm:
                        # 原算法
                        scores = torch.matmul(query_vectors_out, torch.transpose(tool_vectors_out, 0, 1))
                    else:
                        # 考虑模长
                        q_norm_list = [torch.norm(query_vecotr) for query_vecotr in query_vectors_out]
                        t_norm_list = [torch.norm(tool_vector) for tool_vector in tool_vectors_out]
                        norm_matrix = torch.tensor([[((q_norm+t_norm)/2)**2 for t_norm in t_norm_list] for q_norm in q_norm_list]).to(tool_vectors_out.device)
                        scores = torch.matmul(query_vectors_out, torch.transpose(tool_vectors_out, 0, 1)).div(norm_matrix)

                    softmax_scores = F.log_softmax(scores, dim=1)

                    correct_num, all_num = tool_retriever_eval_calculate(softmax_scores.to("cpu"), data["gold_labels"].to("cpu"))
                    eval_counter_dict["correct"] += correct_num
                    eval_counter_dict["all"] += all_num

                    for i in range(len(query_pos_ids)):
                        if config.cal_seq:
                            tool_name_result, _, top_k_name_result = model.tool_selection(data["dataset_name"][i], query_output.hidden_states[-1][i][:query_pos_ids[i]+1].unsqueeze(dim=0))
                        else:
                            tool_name_result, _, top_k_name_result = model.tool_selection(data["dataset_name"][i], query_output.hidden_states[-1][i][query_pos_ids[i]].unsqueeze(dim=0).unsqueeze(dim=0))
                        if data["tool_name"][i] == tool_name_result:
                            eval_counter_dict_alltools["correct"] += 1
                        if data["tool_name"][i] in top_k_name_result:
                            eval_counter_dict_alltools["correct_top_k"] += 1
                    eval_counter_dict_alltools["all"] += len(query_pos_ids)

                logger.info("Train Environment Result: ")
                acc = eval_counter_dict["correct"] / (eval_counter_dict["all"] + 1e-10)
                logger.info('Eval: ACC: {} '.format(str(acc)))

                logger.info("Test Environment Result: ")
                acc_alltools = eval_counter_dict_alltools["correct"] / (eval_counter_dict_alltools["all"] + 1e-10)
                logger.info('Eval: TOP_1 ACC: {} '.format(str(acc_alltools)))
                acc_alltools_top_k = eval_counter_dict_alltools["correct_top_k"] / (eval_counter_dict_alltools["all"] + 1e-10)
                logger.info('Eval: TOP_k ACC: {} '.format(str(acc_alltools_top_k)))                
            if mode == "test":
                break
        if epoch>=3:
            model.sl_retriever(config.model_dir, "save", "epoch_"+str(epoch))

def tool_retriever_eval_calculate(predict_labels, gold_labels):
    predict_labels = predict_labels.tolist()
    gold_labels = gold_labels.tolist()
    correct_counter = 0
    all_counter = len(gold_labels)
    for i in range(all_counter):
        if predict_labels[i].index(max(predict_labels[i])) == gold_labels[i]:
            correct_counter += 1
    return correct_counter, all_counter

def return_retriever_train():
    pass