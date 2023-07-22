import os
from IPython.display import clear_output as clear
import json
import random
import paddle

def get_response(prompt, model, tokenizer, input_length, output_length,print_prompt=True, rand = False):
    if(len(prompt)>input_length):
        prompt = prompt[-input_length:-1]
    if(rand is True):
        temperature = random.random()
        # print(temperature)
    else:
        temperature = 0.5
    if(print_prompt is True):
        print(prompt,end='')
    inputs = tokenizer(
        prompt,
        return_tensors="np",
        padding=True,
        max_length=input_length,
        truncation=True,
        truncation_side="left",
    )
    input_map = {}
    for key in inputs:
        input_map[key] = paddle.to_tensor(inputs[key])
    infer_result = model.generate(
        **input_map,
        decode_strategy="sampling",
        top_k=20 ,
        max_length=output_length,
        # use_cache=True,
        use_fast=True,
        use_fp16_decoding=True,
        repetition_penalty=1.2,
        temperature = temperature,
        length_penalty=1,
    )[0]
    res = tokenizer.decode(infer_result.tolist()[0], skip_special_tokens=True)
    res = res.strip("\n")
    return res
def convert_example(example, tokenizer, get_query=False):
    query = example["user_input"]
    response = example["sys_output"]
    history = example.get("history", None)
    if history is None or len(history) == 0:
        # prompt = query
        prompt = '[Round 0]\n问：{}\n答：'.format(query)
    else:
        prompt = ""
        # for i, (old_query, old_response) in enumerate(history):
        #     prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, old_response)
        if(get_query is False):
            # prompt = ""
            for i, (old_query, old_response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, old_response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        else:
            # prompt = "上下文："
            for i, (old_query, old_response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, old_response)
            # prompt += "\n问：".format(len(history))
            prompt += "[Round {}]\n基于上面的谈话，问：".format(len(history))
    return prompt

def prompt_test()
input_length = 2048  # max input length
output_length = 160  # max output length
start = 0
history = ''
# 括号内前面是用户说，后面是系统答。修改和设计对话历史，让模型顺着前文更好地回答
history = [
    ('“你是一位经验十分丰富的口语对话教练,你的任务是使用英语与他人进行日常对话”。',
     '你好，我是你的对话教练，我们现在开始吧！'),
    ('How are you', 'Fine, thank you,and you?')]
auto_query = '你好，我刚才有点走神，没有听清你的问题，可以再说一遍吗？'
example = {'user_input': '', 'sys_output': '', 'history': history}
# # 载入对话历史
# example = np.load('history_chat.npy',allow_pickle=True).tolist()
# example['user_input']=''
print(convert_example(example, tokenizer, get_query=False))
while (1):
    user_input = input()
    if (user_input == ''):
        user_input = auto_query
    if (user_input == '1'):
        print('-----\n》》退出')
        break
    if (user_input == '2'):
        # prompt = convert_example_neko(example, tokenizer, get_query=True)
        prompt = convert_example(example, tokenizer, get_query=True)
        auto_query = get_response(prompt, model, tokenizer, input_length, output_length, print_prompt=False, rand=True)
        print('-----\n》》是否计划回复（是则直接回车，否则不用管）：' + auto_query)
        continue

    if (user_input == '0'):
        user_input_pre = example['history'][-1][0]
        example['history'].pop()
        # prompt = convert_example_neko(example, tokenizer, get_query=False)
        prompt = convert_example(example, tokenizer, get_query=False)
        os.system('cls' if os.name == 'nt' else 'clear')
        clear()
        response = get_response(prompt, model, tokenizer, input_length, output_length, print_prompt=True, rand=True)
        print(response)
        example['history'].append((user_input_pre, response))
        print('-----\n》》直接输入进行回复；或者扣1退出；扣2提示一个回复例子;扣9保存对话历史；扣0换一个系统回答；')
        continue
    if (user_input == '3'):
        user_input = '你好，我刚才有点走神，没有听清你的问题，可以再说一遍吗？'
    if (user_input == '9'):
        import numpy as np

        np.save('history_chat', example)
        print('我们之前的悄悄话已保存在 history_chat.npy，请不要偷看！')
        continue

    os.system('cls' if os.name == 'nt' else 'clear')
    clear()
    example['user_input'] = user_input
    # prompt = convert_example_neko(example, tokenizer, get_query=False)
    prompt = convert_example(example, tokenizer, get_query=False)
    res = get_response(prompt, model, tokenizer, input_length, output_length, print_prompt=True, rand=False)

    example['sys_output'] = res
    example['history'].append((user_input, res))

    print(res)
    # 遗忘
    while (len(convert_example(example, tokenizer, get_query=False)) + 16 >= input_length):
        example['history'] = example['history'][1:]
    print('-----\n》》直接输入进行回复；或者扣1退出；扣2提示一个回复例子;扣9保存对话历史；扣0换一个系统回答；')