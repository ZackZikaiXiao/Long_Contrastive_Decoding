import jsonlines
import random 
import os
import numpy as np
import math
import argparse



def build_kv_retrieval(number_of_sample, number_of_noise, length, source_file, save_dir):

    target_length = [64 * 1024, 128 * 1024]
    # interv = [16, 7]
    # nsample = [500, 500]
    # nnoise = [0, 75*1]    4k
    nsample = [300]
    nsample.append(number_of_sample)
    nnoise = [0]
    nnoise.append(number_of_noise)
    
    for ii in range(1, 2):
        cnt = -1
        ret = []

        with jsonlines.open(source_file) as fin:
            for line in fin:
                # print(len(line["ordered_kv_records"]))
                # return 0
                cnt += 1
                if cnt == nsample[ii]:
                    break
                ans_id = min(int(cnt * nnoise[ii] / nsample[ii]), nnoise[ii])

                text = "JSON data:\n{"
                t = -1
                random.shuffle(line["ordered_kv_records"])
                for item in line["ordered_kv_records"]:
                    t += 1
                    if t == nnoise[ii]:
                        break
                    text += "\"" + item[0] + "\": \"" + item[1] + "\", "
                text = text[:-2] + '}'
                question = "\nKey: \"" + line["ordered_kv_records"][ans_id][0] +  "\"\nThe value associated with the specified key is: "
                # text += "\nKey: \"" + line["ordered_kv_records"][ans_id][0] +  "\"\nThe value associated with the specified key is: "
                # print(len(tokenizer.encode(text)))
                # break
                ret.append({"id": cnt, "ans_id": ans_id, "context": text, "input": question, "answer": line["ordered_kv_records"][ans_id][1]})
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fw = jsonlines.open(save_dir + "/" + "kv_retrieval_noise_" + str(length) + ".jsonl", 'w')
        fw.write_all(ret)
        fw.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build KV Retrieval Data')
    
    parser.add_argument('--source_file', type=str, default='./benchmark/super_retrieval/kv-retrieval-3000_keys.jsonl', help='Path to the source JSONL file')
    parser.add_argument('--save_dir', type=str, default='./motivation/data', help='Directory to save the output files')

    args = parser.parse_args()

    # 已知的长度和 nnoise 值
    length_4k = 4000
    nnoise_4k = 75
    ratio = nnoise_4k / length_4k
    
    # 对数刻度下的已知值
    log_nnoise_4k = np.log10(nnoise_4k)
    log_length_4k = np.log10(length_4k)


    # 定义16个点对应的长度范围
    log_min = math.log10(math.pow(2, 10))  # 对应于1k
    log_max = math.log10(math.pow(2, 17))  # 对应于128k
    log_lengths_list = np.linspace(log_min, log_max, 16)

    # 计算对应的长度值
    lengths_list = np.round(10**log_lengths_list).astype(int)
    
    # 通过比例计算对应的 nnoise 值
    nnoise_list = np.round(lengths_list * ratio).astype(int)

    # 输出结果
    print("对应长度的 nnoise 值为：")
    for i, nnoise in enumerate(nnoise_list):
        print(f"点 {i+1}: nnoise = {nnoise}")
        
    print("对应 nnoise 的长度值为：")
    for i, length in enumerate(lengths_list):
        print(f"点 {i+1}: length = {length}")
    
    for i in range(len(log_lengths_list)):    
        build_kv_retrieval(
                           number_of_sample = 100,
                           number_of_noise= nnoise_list[i], 
                           length=lengths_list[i], 
                           source_file="./benchmark/super_retrieval/kv-retrieval-3000_keys.jsonl", 
                           save_dir="./motivation/data"
                           )

    for i in range(len(log_lengths_list)):    
        build_kv_retrieval(
            number_of_sample=100,
            number_of_noise=nnoise_list[i], 
            length=lengths_list[i], 
            source_file=args.source_file, 
            save_dir=args.save_dir
        )
