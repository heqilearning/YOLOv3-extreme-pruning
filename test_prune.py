from models import *
from utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from test import evaluate
from terminaltables import AsciiTable
import time
from utils.prune_utils import *

class opt():
    model_def = "config/prune_0.1_yolov3-hand.cfg"
    data_config = "config/oxfordhand.data"
    model = 'weights/prune_0.1_yolov3_ckpt_39_08210417.pth'


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(opt.model_def).to(device)
model.load_state_dict(torch.load(opt.model))


data_config = parse_data_config(opt.data_config)
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])

eval_model = lambda model:evaluate(model, path=valid_path, iou_thres=0.5, conf_thres=0.01,
    nms_thres=0.5, img_size=model.img_size, batch_size=8)
obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

# origin_model_metric = eval_model(model)
origin_nparameters = obtain_num_parameters(model)

# CBL_idx, Conv_idx, prune_idx= parse_module_defs(model.module_defs)
CBL_idx, Conv_idx = parse_module_defs(model.module_defs)


# bn_weights = gather_bn_weights(model.module_list, prune_idx)
bn_weights = gather_bn_weights(model.module_list, CBL_idx)


sorted_bn = torch.sort(bn_weights)[0]

# 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
highest_thre = []
# for idx in prune_idx:
for idx in CBL_idx:
    highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
highest_thre = min(highest_thre)

# 找到highest_thre对应的下标对应的百分比
percent_limit = (sorted_bn==highest_thre).nonzero().item()/len(bn_weights)
zero_percent = bn_weights.le(0.0001).float().sum().item() / len(bn_weights)

print(f'Threshold should be less than {highest_thre:.4f}.')
print(f'The corresponding prune ratio is {percent_limit:.3f}.')
print(f'The le(0.0001)_gamma percent is {zero_percent:.5f}')

#%%
def prune_and_eval(model, sorted_bn, percent=.0):
    model_copy = deepcopy(model)
    thre_index = int(len(sorted_bn) * percent)
    thre = sorted_bn[thre_index]

    print(f'Channels with Gamma value less than {thre:.4f} are pruned!')

    remain_num = 0
    # for idx in prune_idx:    
    for idx in CBL_idx:

        bn_module = model_copy.module_list[idx][1]

        mask = obtain_bn_mask(bn_module, thre)

        remain_num += int(mask.sum())
        bn_module.weight.data.mul_(mask)

    # mAP = eval_model(model_copy)[2].mean()

    print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
    print(f'Prune ratio: {1-remain_num/len(sorted_bn):.3f}')
    # print(f'mAP of the pruned model is {mAP:.4f}')

    return thre

percent = 0.1
threshold = prune_and_eval(model, sorted_bn, percent)
#%%
def obtain_filters_mask(model, thre, CBL_idx):

    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    remain_mask = []
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        # if idx in prune_idx:

        mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
        remain = int(mask.sum())
        pruned = pruned + mask.shape[0] - remain

        if remain == 0:
            print("Channels would be all pruned!")
            raise Exception

        print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
              f'remaining channel: {remain:>4d}')
    
        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.copy())
        remain_mask.append(mask.nonzero()[0])

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask, remain_mask

num_filters, filters_mask, remain_mask = obtain_filters_mask(model, threshold, CBL_idx)

#%%
CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}

pruned_model = prune_model_keep_size(model, CBL_idx, CBLidx2mask)

# eval_model(pruned_model)

input_masks = {}
for idx in CBL_idx + Conv_idx:
    input_mask = get_input_mask(pruned_model.module_defs, idx, CBLidx2mask)
    input_masks[idx] = input_mask
    pruned_model.module_defs[idx]['in_channels'] = len(input_mask.nonzero()[0])


#%%
compact_module_defs = deepcopy(pruned_model.module_defs)
for idx, num, mask in zip(CBL_idx, num_filters, remain_mask):
    assert compact_module_defs[idx]['type'] == 'convolutional'
    if int(compact_module_defs[idx]['filters']) > num:
        compact_module_defs[idx]['filters'] = str(num)
        if isinstance(compact_module_defs[idx]['remain'], int):
            compact_module_defs[idx]['remain'] = mask
        else:
            compact_module_defs[idx]['remain'] = np.array(compact_module_defs[idx]['remain'])[mask]


for module_def in compact_module_defs:
    if module_def['type'] == 'shortcut':
        if 'remain2' in module_def:
            module_def['remain'] = module_def['remain2']

#%%
compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
compact_nparameters = obtain_num_parameters(compact_model)

# init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)
init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask, input_masks)


#%%
random_input = torch.rand((1, 3, model.img_size, model.img_size)).to(device)

def obtain_avg_forward_time(input, model, repeat=200):

    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output

# pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
# compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

# diff = (pruned_output-compact_output).abs().gt(0.001).sum().item()
# if diff > 0:
#     print('Something wrong with the pruned model!')

#%%
# 在测试集上测试剪枝后的模型, 并统计模型的参数数量
# compact_model_metric = eval_model(compact_model)

# #%%
# # 比较剪枝前后参数数量的变化、指标性能的变化
# metric_table = [
#     ["Metric", "Before", "After"],
#     ["mAP", f'{origin_model_metric[2].mean():.6f}', f'{compact_model_metric[2].mean():.6f}'],
#     ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
#     ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
# ]
# print(AsciiTable(metric_table).table)

#%%
# 生成剪枝后的cfg文件并保存模型
pruned_cfg_name = opt.model_def.replace('/', f'/prune_{percent}_')
pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
print(f'Config file has been saved: {pruned_cfg_file}')

compact_model_name = opt.model.replace('/', f'/prune_{percent}_')
torch.save(compact_model.state_dict(), compact_model_name)
print(f'Compact model has been saved: {compact_model_name}')
