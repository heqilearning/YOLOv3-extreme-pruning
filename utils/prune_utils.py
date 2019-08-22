import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    print('sparse leanrning', sr)
    return sr


def parse_module_defs(module_defs):

    CBL_idx = []
    Conv_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
    return CBL_idx, Conv_idx



def gather_bn_weights(module_list, prune_idx):

    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


def write_cfg(cfg_file, module_defs):

    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    if key in ('input_mask', 'remain2'):
                        continue
                    if key == 'remain' and not isinstance(value, int):
                        value = ' '.join([str(i) for i in value])
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx):
        if sr_flag:
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1


def obtain_quantiles(bn_weights, num_quantile=5):

    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total//num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile+1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def get_input_mask(module_defs, idx, CBLidx2mask):

    if idx == 0:
        return np.ones(3)

    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        # return CBLidx2mask[idx - 2]
        from_layer = int(module_defs[idx - 1]['from'])
        if isinstance(module_defs[idx - 1 + from_layer]['remain'], int) and isinstance(module_defs[idx - 2]['remain'], int):
            if module_defs[idx - 1 + from_layer]['type'] == 'shortcut':
                mask = CBLidx2mask[idx - 2] + module_defs[idx - 1 + from_layer]['input_mask']
            else:
                mask = CBLidx2mask[idx - 2] + CBLidx2mask[idx - 1 + from_layer]
            module_defs[idx - 1]['input_mask'] = mask.copy()
            in_channel_idx = np.argwhere(mask)[:, 0].tolist()
            module_defs[idx - 1]['remain2'] = in_channel_idx
            return mask
        else:
            if isinstance(module_defs[idx - 2]['remain'], int):
                index_1 = list(range(module_defs[idx - 2]['old_filters']))
            else:
                index_1 = module_defs[idx - 2]['remain']
            if isinstance(module_defs[idx - 1 + from_layer]['remain'], int):
                index_2 = list(range(module_defs[idx - 1 + from_layer]['old_filters']))
            else:
                index_2 = module_defs[idx - 1 + from_layer]['remain']
            index_all = module_defs[idx - 1]['remain']
            mask = np.zeros(module_defs[idx - 1]['old_filters'])
            mask[index_1] = mask[index_1] + CBLidx2mask[idx - 2]
            if module_defs[idx - 1 + from_layer]['type'] == 'shortcut':
                mask[index_2] = mask[index_2] + module_defs[idx - 1 + from_layer]['input_mask']
            else:
                mask[index_2] = mask[index_2] + CBLidx2mask[idx - 1 + from_layer]
            mask = mask[index_all]
            module_defs[idx - 1]['input_mask'] = mask.copy()
            in_channel_idx = np.argwhere(mask)[:, 0].tolist()
            module_defs[idx - 1]['remain2'] = np.array(index_all)[in_channel_idx].tolist()
            return mask


    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))
        if len(route_in_idxs) == 1:
            return CBLidx2mask[route_in_idxs[0]]
        elif len(route_in_idxs) == 2:
            return np.concatenate([CBLidx2mask[route_in_idxs[0] - 1], module_defs[route_in_idxs[1]]['input_mask']])

        else:   
            print("Something wrong with route module!")
            raise Exception


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask, input_masks):

    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = input_masks[idx]
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()


    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = input_masks[idx]
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data   = loose_conv.bias.data.clone()



def update_activation(i, pruned_model, activation, CBL_idx):
    next_idx = i + 1
    if pruned_model.module_defs[next_idx]['type'] == 'convolutional':
        next_conv = pruned_model.module_list[next_idx][0]
        conv_sum = next_conv.weight.data.sum(dim=(2, 3))
        offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
        if next_idx in CBL_idx:
            next_bn = pruned_model.module_list[next_idx][1]
            next_bn.running_mean.data.sub_(offset)
        else:
            next_conv.bias.data.add_(offset)


def prune_model_keep_size(model, CBL_idx, CBLidx2mask):

    pruned_model = deepcopy(model)
    activations = []
    for i, model_def in enumerate(model.module_defs):

        if model_def['type'] == 'convolutional':
            activation = None
            if i in CBL_idx:
                mask = torch.from_numpy(CBLidx2mask[i]).cuda()
                bn_module = pruned_model.module_list[i][1]
                bn_module.weight.data.mul_(mask)
                activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
                update_activation(i, pruned_model, activation, CBL_idx)
                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        if model_def['type'] == 'shortcut':
            if isinstance(model_def['remain'], int):
                actv1 = activations[i - 1]
                from_layer = int(model_def['from'])
                actv2 = activations[i + from_layer]
                activation = actv1 + actv2
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)
            else:
                prev_remain = np.array(model_def['remain'])
                prev_1_remain = np.array(model.module_defs[i-1]['remain']) if not isinstance(model.module_defs[i-1]['remain'],int)\
                                                                            else np.arange(model_def['old_filters'])
                from_layer = int(model_def['from'])
                prev_2_remain = np.array(model.module_defs[i+from_layer]['remain']) if not isinstance(model.module_defs[i+from_layer]['remain'],int)\
                                                                            else np.arange(model_def['old_filters'])
                activation = torch.zeros(model_def['old_filters']).cuda()
                activation[prev_1_remain] = activation[prev_1_remain] + activations[i-1]
                activation[prev_2_remain] = activation[prev_2_remain] + activations[i+from_layer]
                activation = activation[prev_remain]
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)


        if model_def['type'] == 'route':
            from_layers = [int(s) for s in model_def['layers'].split(',')]
            if len(from_layers) == 1:
                activation = activations[i + from_layers[0]]
                update_activation(i, pruned_model, activation, CBL_idx)
            else:
                actv1 = activations[i + from_layers[0]]
                actv2 = activations[from_layers[1]]
                activation = torch.cat((actv1, actv2))
                update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        if model_def['type'] == 'upsample':
            activation = activations[i - 1]
            update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        if model_def['type'] == 'yolo':
            activations.append(None)
       
    return pruned_model





def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask
