from datetime import datetime
import os
import torch
from tqdm import tqdm
from global_util import *

DATA_PATH = './data/'
USER_CACHE = True
MODEL_FILE_SUFFIX = '.pt'
CONTENT = 'text'
MIN_LEN = 1
MAX_LEN = 99999


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


def generate_label(filename):
    pt_filename = 'label'
    columns = {'信息ID': 'id', '纯文本正文': 'text', '中标商': 'vendor', '中标金额': 'money', '代理机构': 'agent', '业主': 'owner', 'html正文': 'html'}

    dataset = load_xlsx(filename, pt_filename, columns)

    text_list = dataset[CONTENT]
    owner_list = dataset['owner']
    agent_list = dataset['agent']
    vendor_list = dataset['vendor']
    money_list = dataset['money']

    csv = []

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
        if is_not_none(owner, agent):
            owner = owner.strip()
            agent = agent.strip()
            if text.count(owner) == 1 and text.count(agent) == 1:
                owner_offset = text.find(owner)
                agent_offset = text.find(agent)

                data = {}
                data['sentence'] = text
                data['relation'] = '业主-代理' if owner_offset < agent_offset else '代理-业主'
                data['head'] = owner
                data['head_offset'] = owner_offset
                data['tail'] = agent
                data['tail_offset'] = agent_offset
                csv.append(data)
            else:
                write_log('#%d owner agent has multiple values' % index)
        else:
            write_log('#%d owner agent has None value' % index)

        # 业主-供应商、供应商-业主
        if is_not_none(owner, vendor):
            owner = owner.strip()
            vendor = vendor.strip()
            if text.count(owner) == 1 and text.count(vendor) == 1:
                owner_offset = text.find(owner)
                vendor_offset = text.find(vendor)

                data = {}
                data['sentence'] = text
                data['relation'] = '业主-供应商' if owner_offset < vendor_offset else '供应商-业主'
                data['head'] = owner
                data['head_offset'] = owner_offset
                data['tail'] = vendor
                data['tail_offset'] = vendor_offset
                csv.append(data)
            else:
                write_log('#%d owner vendor has multiple values' % index)
        else:
            write_log('#%d owner vendor has None value' % index)

        # 代理-供应商、供应商-代理
        if is_not_none(agent, vendor):
            agent = agent.strip()
            vendor = vendor.strip()
            if text.count(agent) == 1 and text.count(vendor) == 1:
                agent_offset = text.find(agent)
                vendor_offset = text.find(vendor)

                data = {}
                data['sentence'] = text
                data['relation'] = '代理-供应商' if agent_offset < vendor_offset else '供应商-代理'
                data['head'] = agent
                data['head_offset'] = agent_offset
                data['tail'] = vendor
                data['tail_offset'] = vendor_offset
                csv.append(data)
            else:
                write_log('#%d agent vendor has multiple values' % index)
        else:
            write_log('#%d agent vendor has None value' % index)

    print('dev')


if __name__ == '__main__':
    start_time = datetime.now()
    write_log('start label generating ...')

    # generate_org_label()

    # merge('label/cops_names_bidnotice.NER', 'label/cops_names_bidresult.NER')

    filename = '/Users/caiwei/Documents/Document/招标网/事件抽取/阿里标记数据-01捷风.xlsx'
    generate_label(filename)

    write_log('label generating cost: %s' % (datetime.now() - start_time))
