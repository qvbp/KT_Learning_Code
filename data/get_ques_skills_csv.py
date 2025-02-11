import csv

# 输入文件路径和输出文件路径
input_file = './peiyou/data.txt'  # 替换为您的输入文件路径
output_file = './peiyou/ques_skills.csv'

# 读取文件并处理数据
def parse_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
        
        # 将数据划分为5行一组
        num_lines_per_group = 6
        ques_skill_pairs = []
        all_questions = []  # 用于记录所有问题（不去重）
        unique_questions = set()  # 用于记录唯一问题（去重）
        
        # 逐步解析每5行一组的数据
        for i in range(0, len(lines), num_lines_per_group):
            if i + num_lines_per_group > len(lines):
                break  # 确保行数足够
            
            # 解析每一组的5行数据
            user_id, sequence_length = lines[i].strip().split(',')
            sequence_length = int(sequence_length)  # 转换为整数
            
            # 跳过 sequence_length < 3 的组
            if sequence_length < 3:
                continue
            
            question_ids = lines[i+1].strip().split(',')
            skill_ids = lines[i+2].strip().split(',')
            
            # 记录问题
            all_questions.extend(question_ids)
            unique_questions.update(question_ids)
            
            # 生成题目与知识点的映射关系，避免重复
            for ques_id, skill_id in zip(question_ids, skill_ids):
                skills = skill_id.split('_')  # 拆分知识点
                for skill in skills:
                    pair = (ques_id, skill)
                    if pair not in ques_skill_pairs:
                        ques_skill_pairs.append(pair)

    # 将结果写入CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['ques', 'skill'])
        csv_writer.writerows(ques_skill_pairs)
    
    # 输出问题的数量
    print(f"不去重的问题总数：{len(all_questions)}")
    print(f"去重的问题总数：{len(unique_questions)}")

# 执行函数
parse_file(input_file, output_file)
print(f"数据已成功保存到 {output_file}")
