import csv
from datetime import datetime
import os
import random
import torch
from tqdm import tqdm
from global_util import *

DATA_PATH = './data/'
USER_CACHE = True
MODEL_FILE_SUFFIX = '.pt'
CONTENT = 'text'
MIN_LEN = 1
MAX_LEN = 99999
TRAIN_SIZE = 9
TEST_SIZE = 0.05
DEV_SIZE = 0.05
random.seed(1)


def load_xlsx(filename, pt_filename=None, columns=None):
    """
    加载原始数据并缓存
    """
    pt_filename = DATA_PATH + pt_filename + MODEL_FILE_SUFFIX

    if USER_CACHE and os.path.exists(pt_filename):
        dataset = torch.load(pt_filename)
    else:
        dataset = get_dict_from_xlsx(filename, columns)
        torch.save(dataset, pt_filename)

    return dataset


def is_not_none(a, b):
    return a is not None and b is not None


def add_label(labels, text, index, head, tail, head_tail, tail_head):
    if is_not_none(head, tail):
        head = head.strip()
        tail = tail.strip()

        if text.count(head) == 1 and text.count(tail) == 1:
            head_offset = text.find(head)
            tail_offset = text.find(tail)

            label = {}
            label['sentence'] = text
            label['relation'] = head_tail
            label['head'] = head
            label['head_offset'] = head_offset
            label['tail'] = tail
            label['tail_offset'] = tail_offset
            labels.append(label)

            label = {}
            label['sentence'] = text
            label['relation'] = tail_head
            label['head'] = tail
            label['head_offset'] = tail_offset
            label['tail'] = head
            label['tail_offset'] = head_offset
            labels.append(label)
        else:
            write_log('#%d %s has multiple values' % (index, head_tail))
    else:
        write_log('#%d %s has None value' % (index, head_tail))


def generate_labels(dataset, filename):
    labels_filename = DATA_PATH + filename + '_label.pt'
    if USER_CACHE and os.path.exists(labels_filename):
        return torch.load(labels_filename)

    text_list = dataset[CONTENT]
    owner_list = dataset['owner']
    agent_list = dataset['agent']
    vendor_list = dataset['vendor']
    money_list = dataset['money']

    labels = []

    for index in tqdm(range(len(text_list))):
        text = text_list[index]
        owner = owner_list[index]
        agent = agent_list[index]
        vendor = vendor_list[index]
        # money = money_list[index]

        if text is None:
            continue
        else:
            text = text.strip()
            if len(text) < MIN_LEN or len(text) > MAX_LEN:
                continue

        # 业主-代理、代理-业主
        add_label(labels, text, index, owner, agent, '业主-代理', '代理-业主')

        # 业主-供应商、供应商-业主
        add_label(labels, text, index, owner, vendor, '业主-供应商', '供应商-业主')

        # 代理-供应商、供应商-代理
        add_label(labels, text, index, agent, vendor, '代理-供应商', '供应商-代理')

    torch.save(labels, labels_filename)

    return labels


def spllit_labels(labels):
    random.shuffle(labels)

    size = len(labels)

    train_size = int(size * TRAIN_SIZE / (TRAIN_SIZE + DEV_SIZE + TEST_SIZE))
    dev_size = int(size * DEV_SIZE / (TRAIN_SIZE + DEV_SIZE + TEST_SIZE))

    train_labels = labels[:train_size]
    dev_labels = labels[train_size:train_size + dev_size]
    test_labels = labels[train_size + dev_size:]

    return train_labels, dev_labels, test_labels


def save_csv(labels, filename):
    columns = ['sentence', 'relation', 'head', 'head_offset', 'tail', 'tail_offset']

    with open(DATA_PATH + filename, 'w') as file:
        csv_file = csv.writer(file)
        csv_file.writerow(columns)
        for label in labels:
            row = [label['sentence'], label['relation'], label['head'], label['head_offset'], label['tail'], label['tail_offset']]
            csv_file.writerow(row)

    print('dev')


def generate_label(filename):
    pt_filename = 'label'
    columns = {'信息ID': 'id', '纯文本正文': 'text', '中标商': 'vendor', '中标金额': 'money', '代理机构': 'agent', '业主': 'owner', 'html正文': 'html'}

    dataset = load_xlsx(filename, pt_filename, columns)

    labels = generate_labels(dataset, pt_filename)

    train_labels, dev_labels, test_labels = spllit_labels(labels)

    save_csv(train_labels, 'train.csv')
    save_csv(dev_labels, 'dev.csv')
    save_csv(test_labels, 'test.csv')


if __name__ == '__main__':
    start_time = datetime.now()
    write_log('start label generating ...')

    # generate_org_label()

    # merge('label/cops_names_bidnotice.NER', 'label/cops_names_bidresult.NER')

    filename = '/Users/caiwei/Documents/Document/招标网/事件抽取/阿里标记数据-01捷风.xlsx'
    generate_label(filename)

    write_log('label generating cost: %s' % (datetime.now() - start_time))
