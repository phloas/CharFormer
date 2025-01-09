import xml.etree.ElementTree as ET
from xml import etree
import re

from Bio import SeqIO
 
# 打开XML文件进行迭代解析
# context = ET.iterparse('.data/DBLP/dblp.xml', events=('start',))
# context = ET.iterparse('.data/uniref/uniref50/uniref50.xml', events=('start',))

## 处理uniref50.fasta文件
# with open('.data/uniref/uniref50/uniref50.fasta','rt') as f:
#     lines = f.readlines()

# protein_sequences = []
# current_sequence = ""
# # 遍历每一行
# for line in lines:
    
#     line = line.strip()
#     print(line)
#     if line.startswith('>'):
#         # 如果是标题行，则保存当前序列并开始新的序列
#         if current_sequence:
#             protein_sequences.append(current_sequence)
#             current_sequence = ""
#         if len(protein_sequences) >= 1:
#             break
#     else:
#         # 如果不是标题行，则将行添加到当前序列中
#         current_sequence += line

# # 保存最后一个序列
# if current_sequence:
#     protein_sequences.append(current_sequence)

# # 将蛋白质序列写入文件，每个序列占据一行
# with open('protein_sequences.txt', 'w') as f:
#     for sequence in protein_sequences:
#         f.write(sequence + '\n')
        

def read_DBLP():
    # 处理DBLP数据
    xml_file = '.data/DBLP/dblp.xml'
    parser = etree.XMLParser(dtd_validation=True)
    tree = etree.parse(xml_file, parser=parser)
    root = tree.getroot()
    count = 0
    with open('.data/DBLP/data.txt','w') as f:  
        for elem in root:
            for sub_elem in elem:
                if sub_elem.tag == 'title':
                    title = sub_elem.text
                    f.write(str(sub_elem.text) + '\n')
                    count += 1
            if count >= 200000:
                break


## 处理uniref数据集
# xml_file = '/home/phloas/workspace/archive_project/CharTransformer/.data/uniref/uniref50/uniref50.xml'

# # 打开 XML 文件并创建解析器
# context = ET.iterparse(xml_file, events=("start", "end"))

# # 遍历解析器迭代器
# with open('/home/phloas/workspace/archive_project/CharTransformer/.data/uniref/uniref50/data.txt','w') as f:
#     for event, elem in context:
#         count = 0  
#         # 在每个元素结束时处理数据
#         if event == "end":
#             f.write(str(elem.tag) + str(elem.attrib)+'\n')
#             count += 1
#             # 在此处处理元素数据，例如提取所需信息、存储到数据库等
#             # 这里的 elem 表示当前处理的 XML 元素
            
#             # 清理当前元素以释放内存
#             elem.clear()
#         if count >= 500:
#             break

def read_sequence():
    uniref50_file_path = '.data/uniref/uniref50/uniref50.fasta'
    uniref90_file_path = '.data/uniref/uniref90/uniref90.fasta'
    # sequencs = SeqIO.parse(uniref_file_path, 'fasta')

    count = 0
    # with open(".data/uniref/uniref50/protein_sequences.txt", "w") as output_file:
    #     for record in SeqIO.parse(uniref50_file_path, 'fasta'):
    #         # if len(record.seq) > 5000:
    #         #     record.seq = record.seq[:5000]
    #         output_file.write(str(record.seq) + '\n')
    #         count += 1
    #         if count >= 100000:
    #             break

    with open(".data/uniref/uniref90/protein_sequences_article.txt", "w") as output_file:
        for record in SeqIO.parse(uniref90_file_path, 'fasta'):
            if len(record.seq) >= 200 and len(record.seq) <= 35213:
                # record.seq = record.seq[:5000]
                output_file.write(str(record.seq) + '\n')
            # count += 1
            # if count >= 100000:
            #     break
               
def read_uniref50():
    source_organisms = []
    
    with open('.data/uniref/uniref50/data.txt','r') as f:
        for line in f:
            match = re.search(r"{'type': 'source organism', 'value': '(.*?)'}", line)
            if match:
                source_organisms.append(match.group(1))
        

    with open(".data/uniref/uniref50/source_organisms.txt", "w") as output_file:
        for name in source_organisms:
            output_file.write(name + "\n")


def remove_duplicate_lines(input_file, output_file):
    unique_lines = set()
    with open(input_file, "r") as file:
        for line in file:
            unique_lines.add(line.strip())  # 去除行末尾的换行符并添加到集合中

    with open(output_file, "w") as file:
        for line in unique_lines:
            file.write(line + "\n")  # 将唯一行写入新文件中



if __name__ == '__main__':

    # read_uniref()
    # remove_duplicate_lines(".data/uniref/uniref50/source_organisms.txt", ".data/uniref/uniref50/source_organism.txt")
    read_sequence()