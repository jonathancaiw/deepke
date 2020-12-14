import logging
import os
import random
import torch
from tqdm import tqdm
import hydra
import models
from hydra import utils
from utils import load_pkl, load_csv
from serializer import Serializer
from preprocess import _serialize_sentence, _convert_tokens_into_index, _add_pos_seq, _handle_relation_data
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _preprocess_data(data, cfg):
    vocab = load_pkl(os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl'), verbose=False)
    relation_data = load_csv(os.path.join(cfg.cwd, cfg.data_path, 'relation.csv'), verbose=False)
    rels = _handle_relation_data(relation_data)
    cfg.vocab_size = vocab.count
    serializer = Serializer(do_chinese_split=cfg.chinese_split)
    serial = serializer.serialize

    _serialize_sentence(data, serial, cfg)
    _convert_tokens_into_index(data, vocab)
    _add_pos_seq(data, cfg)
    logger.info('start sentence preprocess...')
    formats = '\nsentence: {}\nchinese_split: {}\nreplace_entity_with_type:  {}\nreplace_entity_with_scope: {}\n' \
              'tokens:    {}\ntoken2idx: {}\nlength:    {}\nhead_idx:  {}\ntail_idx:  {}'
    logger.info(
        formats.format(data[0]['sentence'], cfg.chinese_split, cfg.replace_entity_with_type,
                       cfg.replace_entity_with_scope, data[0]['tokens'], data[0]['token2idx'], data[0]['seq_len'],
                       data[0]['head_idx'], data[0]['tail_idx']))

    return data, rels


def add_instance(instances, label, head, tail, head_tail, tail_head):
    if head not in label or tail not in label:
        return

    text = label['text']
    head_entity = label[head]
    tail_entity = label[tail]

    # 内容长的做为head
    if len(head_entity) < len(tail_entity):
        temp = head_entity
        head_entity = tail_entity
        tail_entity = temp

        temp = head_tail
        head_tail = tail_head
        tail_head = temp

    head_pos = get_pos(text, head_entity)
    tail_pos = get_pos(text, tail_entity)

    for head_index in head_pos:
        for tail_index in tail_pos:
            if tail_index in head_pos:
                continue
            else:
                instance = {}
                instance['sentence'] = text
                instance['head'] = head_entity
                instance['tail'] = tail_entity
                instance['head_type'] = '组织机构'
                instance['tail_type'] = '组织机构'
                instance['head_offset'] = str(head_index)
                instance['tail_offset'] = str(tail_index)
                instance['relation'] = head_tail
                instances.append(instance)

                reversed_instance = {}
                reversed_instance['sentence'] = text
                reversed_instance['head'] = tail_entity
                reversed_instance['tail'] = head_entity
                reversed_instance['head_type'] = '组织机构'
                reversed_instance['tail_type'] = '组织机构'
                reversed_instance['head_offset'] = str(tail_index)
                reversed_instance['tail_offset'] = str(head_index)
                reversed_instance['relation'] = tail_head
                instances.append(reversed_instance)


def get_pos(text, entity):
    entity_pos = []

    pos = 0
    while True:
        index = text.find(entity, pos)
        if index > -1:
            entity_pos.append(index)
            pos = index + 1
        else:
            break

    return entity_pos


def _get_predict_instances(cfg):
    labels = torch.load(os.path.join(cfg.cwd, cfg.data_path, 'label_relation.pt'))
    instances = []

    for index in tqdm(range(len(labels))):
        label = labels[index]
        add_instance(instances, label, 'owner', 'agent', '业主-代理', '代理-业主')
        add_instance(instances, label, 'owner', 'vendor', '业主-供应商', '供应商-业主')
        add_instance(instances, label, 'agent', 'vendor', '代理-供应商', '供应商-代理')

    return instances


# 自定义模型存储的路径
fp = '/Users/caiwei/Documents/PycharmProjects/relation_extraction/checkpoints/cnn_complex_epoch10_t0.9_l256.pth'


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    cfg.pos_size = 2 * cfg.pos_limit + 2
    print(cfg.pretty())

    # get predict instance
    instances = _get_predict_instances(cfg)
    random.shuffle(instances)
    data = instances[:1000]

    # preprocess data
    data, rels = _preprocess_data(data, cfg)

    # model
    __Model__ = {
        'cnn': models.PCNN,
        'rnn': models.BiLSTM,
        'transformer': models.Transformer,
        'gcn': models.GCN,
        'capsule': models.Capsule,
        'lm': models.LM,
    }

    # 最好在 cpu 上预测
    # cfg.use_gpu = False
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')
    logger.info(f'device: {device}')

    model = __Model__[cfg.model_name](cfg)
    logger.info(f'model name: {cfg.model_name}')
    logger.info(f'\n {model}')
    model.load(fp, device=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        match = 0

        for sample in tqdm(data):
            x = dict()
            x['word'], x['lens'] = torch.tensor([sample['token2idx']]), torch.tensor([sample['seq_len']])
            if cfg.model_name != 'lm':
                x['head_pos'], x['tail_pos'] = torch.tensor([sample['head_pos']]), torch.tensor([sample['tail_pos']])
                if cfg.model_name == 'cnn':
                    if cfg.use_pcnn:
                        x['pcnn_mask'] = torch.tensor([sample['entities_pos']])

            for key in x.keys():
                x[key] = x[key].to(device)

            y_pred = model(x)
            y_pred = torch.softmax(y_pred, dim=-1)[0]
            prob = y_pred.max().item()
            prob_rel = list(rels.keys())[y_pred.argmax().item()]
            if sample['relation'] == prob_rel:
                match += 1
            else:
                if prob > 0.9:
                    logger.info(f"\"{sample['head']}\" 和 \"{sample['tail']}\" 在句中关系为：\"{prob_rel}\"，置信度为{prob:.2f}，标签值为{sample['relation']}。")

        logger.info('match %d / total %d / acc %.4f' % (match, len(data), match / len(data)))

    if cfg.predict_plot:
        # maplot 默认显示不支持中文
        plt.rcParams["font.family"] = 'Arial Unicode MS'
        x = list(rels.keys())
        height = list(y_pred.cpu().numpy())
        plt.bar(x, height)
        for x, y in zip(x, height):
            plt.text(x, y, '%.2f' % y, ha="center", va="bottom")
        plt.xlabel('关系')
        plt.ylabel('置信度')
        plt.xticks(rotation=315)
        plt.show()


if __name__ == '__main__':
    main()
