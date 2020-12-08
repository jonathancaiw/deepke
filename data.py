import csv
from datetime import datetime
import os
import random
import shutil
import torch
from tqdm import tqdm
from global_util import *

random.seed(1)

DATA_PATH = './data/origin/'
CACHE_PATH = './data/out/'
USER_CACHE = True
MODEL_FILE_SUFFIX = '.pt'
CONTENT = 'text'
CONTENT_MIN_LEN = 1
CONTENT_MAX_LEN = 1000
ORG_MIN_LEN = 2
TRAIN_SIZE = 0.9
TEST_SIZE = 0.05
DEV_SIZE = 0.05


def load_xlsx(xlsx_filename, pt_filename=None, columns=None):
    """
    加载原始数据并缓存
    """
    pt_filename = DATA_PATH + pt_filename + MODEL_FILE_SUFFIX

    if USER_CACHE and os.path.exists(pt_filename):
        dataset = torch.load(pt_filename)
    else:
        dataset = get_dict_from_xlsx(xlsx_filename, columns)
        torch.save(dataset, pt_filename)

    return dataset


def is_not_none(a, b):
    return a is not None and b is not None


def add_label(labels, text, index, head, tail, head_tail, tail_head):
    if is_not_none(head, tail):
        head = head.strip()
        tail = tail.strip()

        if len(head) < ORG_MIN_LEN or len(tail) < ORG_MIN_LEN:
            return

        # 跳过名称相同的错误标注
        if head == tail:
            write_log('#%d %s has same value' % (index, head_tail))
            return

        contain_check = False
        if head.find(tail) > -1:
            if text.count(head) == 1 and text.count(tail) == 2:
                contain_check = True
            else:
                return
        elif tail.find(head) > -1:
            if text.count(head) == 2 and text.count(tail) == 1:
                contain_check = True
            else:
                return

        if contain_check or (text.count(head) == 1 and text.count(tail) == 1):
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
            if len(text) < CONTENT_MIN_LEN or len(text) > CONTENT_MAX_LEN:
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


def generate_label(dataset, pt_filename):
    labels = generate_labels(dataset, pt_filename)

    save_datasets(labels)


def save_datasets(labels):
    train_labels, dev_labels, test_labels = spllit_labels(labels)

    save_csv(train_labels, 'train.csv')
    save_csv(dev_labels, 'valid.csv')
    save_csv(test_labels, 'test.csv')

    # 生成数据集后删除删除数据集缓存
    if os.path.exists(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)


def generate_complex_labels(dataset, filename):
    labels_filename = DATA_PATH + filename + '_complex.pt'
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
            if len(text) < CONTENT_MIN_LEN:
                continue

        if owner is not None:
            owner = owner.strip()
        if agent is not None:
            agent = agent.strip()
        if vendor is not None:
            vendor = vendor.strip()

        # 跳过名称相同的错误标注
        if owner is not None and owner == agent:
            continue
        elif agent is not None and agent == vendor:
            continue
        elif vendor is not None and vendor == owner:
            continue

        label = {}
        if owner is not None and len(owner) >= ORG_MIN_LEN:
            label['owner'] = owner
        if agent is not None and len(agent) >= ORG_MIN_LEN:
            label['agent'] = agent
        if vendor is not None and len(vendor) >= ORG_MIN_LEN:
            label['vendor'] = vendor

        # 小于两个实体无法构造关系三元组
        if len(label) > 1:
            label['text'] = text
            labels.append(label)

    torch.save(labels, labels_filename)

    return labels


def generate_complex_label(dataset, pt_filename):
    labels = generate_complex_labels(dataset, pt_filename)

    save_datasets(labels)


if __name__ == '__main__':
    start_time = datetime.now()
    write_log('start label generating ...')

    xlsx_filename = '/Users/caiwei/Documents/Document/招标网/事件抽取/阿里标记数据-01捷风.xlsx'
    pt_filename = 'label'
    columns = {'信息ID': 'id', '纯文本正文': 'text', '中标商': 'vendor', '中标金额': 'money', '代理机构': 'agent', '业主': 'owner', 'html正文': 'html'}

    dataset = load_xlsx(xlsx_filename, pt_filename, columns)

    # generate_label(dataset, pt_filename)
    generate_complex_label(dataset, pt_filename)

    write_log('label generating cost: %s' % (datetime.now() - start_time))
