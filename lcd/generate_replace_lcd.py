import transformers
from typing import Callable, List, Optional, Tuple, Union
import torch
import inspect
import warnings
from torch import nn
import torch.nn.functional as F
import numpy as np

from transformers.generation.utils import (
    ModelOutput,
    is_hqq_available,
    is_quanto_available,
    is_torchdynamo_compiling,
)
from transformers.cache_utils import (
    DynamicCache,
    HQQQuantizedCache,
    HybridCache,
    QuantizedCacheConfig,
    QuantoQuantizedCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.streamers import BaseStreamer
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.utils import _split_model_inputs, stack_model_outputs
from transformers.utils import (
    ModelOutput,
    is_hqq_available,
    is_quanto_available,
    is_torchdynamo_compiling,
    logging
)
logger = logging.get_logger(__name__)
NEED_SETUP_CACHE_CLASSES_MAPPING = {"static": StaticCache, "sliding_window": SlidingWindowCache, "hybrid": HybridCache}
QUANT_BACKEND_CLASSES_MAPPING = {"quanto": QuantoQuantizedCache, "HQQ": HQQQuantizedCache}
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
import torch.distributed as dist
from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint


def calculate_kl_divergence(probs_with, probs_without):
    # 在对概率取对数之前，添加一个小的正常数以避免对零取对数
    eps = 1e-9
    probs_with_safe = torch.clamp(probs_with, min=eps)  # 确保概率值不小于eps
    probs_without_safe = torch.clamp(probs_without, min=eps)
    kl_div = F.kl_div(probs_with_safe.log(), probs_without_safe, reduction='batchmean')
    return kl_div.item()


def calculate_absolute_difference(probs_with, probs_without):
    abs_diff = torch.sum(torch.abs(probs_with - probs_without))
    return abs_diff.item()


def generate_long_tail_distribution(top_k, size=1000, distribution='powerlaw'):
    if distribution == 'powerlaw':
        # 使用幂律分布
        a = 1.5  # 幂律指数参数
        x = np.random.power(a, size)
    elif distribution == 'lognormal':
        # 使用对数正态分布
        mu, sigma = 0, 1  # 均值和标准差
        x = np.random.lognormal(mu, sigma, size)
    else:
        raise ValueError("Unsupported distribution type. Use 'powerlaw' or 'lognormal'.")
    
    # 对分布进行排序并取top_k
    x_sorted = np.sort(x)[::-1]
    top_k_values = x_sorted[:top_k]
    
    return top_k_values

def weighted_mapping(w_list, x1, x2):
    min_w = min(w_list)
    max_w = max(w_list)
    # 如果最小值和最大值相等，则直接返回范围的中值
    if min_w == max_w:
        return [0.5 * (x1 + x2)] * len(w_list)
    
    mapped_list = [(x2 - x1) * (w - min_w) / (max_w - min_w) + x1 for w in w_list]
    return mapped_list


def generate_long_tail_list(length, imb_type='exp', imb_factor=0.5):
    """
    生成一个具有长尾分布的列表
    :param length: 列表长度
    :param imb_type: 分布类型，可以是'exp', 'step', 'trunk'
    :param imb_factor: 分布的陡峭程度因子
    :return: 长尾分布的列表
    """
    img_max = length
    img_num_per_cls = []
    
    if imb_type == 'exp':
        for cls_idx in range(length):
            num = img_max * (imb_factor**(cls_idx / (length - 1.0)))
            img_num_per_cls.append(num)
    elif imb_type == 'step':
        for cls_idx in range(length // 2):
            img_num_per_cls.append(img_max)
        for cls_idx in range(length // 2):
            img_num_per_cls.append(img_max * imb_factor)
    elif imb_type == 'trunk':  # imb_factor为截取的比例，如0.8
        img_num_per_cls = [img_max for i in range(length)]
        for i in range(len(img_num_per_cls)):
            img_num_per_cls[i] = imb_factor * img_num_per_cls[i]
    else:
        img_num_per_cls.extend([img_max] * length)

    # 归一化处理，使数据更适合显示
    img_num_per_cls = np.array(img_num_per_cls)
    img_num_per_cls = img_num_per_cls / np.max(img_num_per_cls)
    
    return img_num_per_cls.tolist()


class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


class GenerateEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None

class GenerateBeamDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


class GenerateBeamEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput

ContrastiveSearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput

BeamSearchDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
BeamSampleDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput

BeamSearchEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
BeamSampleEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]

# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]

# def _sample(
#         self,
#         input_ids: torch.LongTensor,
#         logits_processor: LogitsProcessorList,
#         stopping_criteria: StoppingCriteriaList,
#         generation_config: GenerationConfig,
#         synced_gpus: bool,
#         streamer: Optional["BaseStreamer"],
#         logits_warper: Optional[LogitsProcessorList] = None,
#         **model_kwargs,
#     ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

#         # init values
#         pad_token_id = generation_config.pad_token_id
#         output_attentions = generation_config.output_attentions
#         output_hidden_states = generation_config.output_hidden_states
#         output_scores = generation_config.output_scores
#         output_logits = generation_config.output_logits
#         return_dict_in_generate = generation_config.return_dict_in_generate
#         has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
#         do_sample = generation_config.do_sample
#         if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
#             raise ValueError(
#                 "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
#                 f"{logits_warper})."
#             )

#         # init attention / hidden states / scores tuples
#         scores = () if (return_dict_in_generate and output_scores) else None
#         raw_logits = () if (return_dict_in_generate and output_logits) else None
#         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
#         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
#         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

#         # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
#             encoder_hidden_states = (
#                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
#             )

#         # keep track of which sequences are already finished
#         batch_size = input_ids.shape[0]
#         this_peer_finished = False
#         unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
#         model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
#         calibration_enable = True
#         prefilling_stage = True
#         while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
#             if calibration_enable:
#                 # prepare model
#                 # if not first_iteration:
#                 if prefilling_stage:
#                     if input_ids.shape[1] == 1:
#                         raise ValueError("input_ids 只有一个元素，无法执行操作。")
#                     input_ids_next = input_ids[:, -1]
#                     input_ids = input_ids[:, :-1]
#                     model_kwargs['attention_mask'] = model_kwargs['attention_mask'][:, :-1]  
#                     model_kwargs['cache_position'] = model_kwargs['cache_position'][:-1] 
#                     model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    
#                     # model_inputs['input_ids'] = model_inputs['input_ids'][:, :-1]  
#                     # model_inputs['position_ids'] = model_inputs['position_ids'][:, :-1]
#                     # model_inputs['cache_position'] = model_inputs['cache_position'][:-1]
#                     # model_inputs['attention_mask'] = model_inputs['attention_mask'][:, :-1]  
#                     outputs = self(
#                                     **model_inputs,
#                                     return_dict=True,
#                                     output_attentions=output_attentions,
#                                     output_hidden_states=output_hidden_states,
#                     )
#                     input_ids = torch.cat([input_ids, input_ids_next[:, None]], dim=-1)
#                     model_kwargs = self._update_model_kwargs_for_generation(
#                         outputs,
#                         model_kwargs,
#                         is_encoder_decoder=self.config.is_encoder_decoder,
#                     )
#                     prefilling_stage = False
                
#                 self.logits_chunks = []

#                 # permu_num = [1, 10, 100, 1000, 10000, 100000]
#                 permu_num = [100] * 1
#                 # 2. 处理每个块
#                 model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
#                 for num in permu_num:
#                     # 在position_ids上添加扰动
#                     # model_inputs['position_ids'] += torch.randint(num, 10 * num, model_inputs['position_ids'].shape, device=model_inputs['position_ids'].device)
#                     # model_inputs['position_ids'].clamp_(min=0)
#                     # model_inputs['position_ids'].fill_(1)   # 1048576
#                     if synced_gpus and this_peer_finished: 
#                         continue  # don't waste resources running the code we don't need
#                     with torch.no_grad():
#                         outputs_chunk = self(
#                             **model_inputs,
#                             return_dict=True,
#                             output_attentions=output_attentions,
#                             output_hidden_states=output_hidden_states,
#                             update_past_key_values=True
#                         )
#                     # 裁剪kv cache
#                     model_inputs['past_key_values'].crop(model_inputs['attention_mask'].shape[1]-1)
#                     if synced_gpus and this_peer_finished:
#                         continue  # don't waste resources running the code we don't need

#                     # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
#                     # (the clone itself is always small)
#                     next_token_logits = outputs_chunk.logits[:, -1, :].clone()
#                     self.logits_chunks.append(next_token_logits)
#                     del outputs_chunk
                    
#                 # 3. formal next token predict
#                 model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
#                 model_inputs["attention_mask"][0].fill_(1)
#                 with torch.no_grad():
#                     outputs = self(
#                         **model_inputs,
#                         return_dict=True,
#                         output_attentions=output_attentions,
#                         output_hidden_states=output_hidden_states,
#                     )

#                 if synced_gpus and this_peer_finished:
#                     continue  # don't waste resources running the code we don't need

#                 # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
#                 # (the clone itself is always small)
#                 next_token_logits_base = outputs.logits[:, -1, :].clone()                

#                 # 计算每个块的KL散度
#                 # self.w_list = []
#                 # for i in range(len(self.logits_chunks)):
#                 #     self.w_list.append(calculate_absolute_difference(self.logits_chunks[i].squeeze(0), next_token_logits_base.squeeze(0)))
                
#                 # self.w_list = weighted_mapping(self.w_list, 1, 2)
#                 self.logits_list = []

#                 for i in range(len(self.logits_chunks)):
#                     # 只对选出来的topk元素进行计算
#                     # next_token_logits_base_softmax = nn.functional.softmax(next_token_logits_base, dim=-1)
#                     # self.logits_chunks[i] = nn.functional.softmax(self.logits_chunks[i], dim=-1)
#                     calibrated_logits = (1 + 1.5) * next_token_logits_base - 1.5 * self.logits_chunks[i]
#                     # calibrated_logits = next_token_logits_base - 0.1 * self.logits_chunks[i]
#                     # calibrated_logits = next_token_logits_base_softmax - self.logits_chunks[i]
#                     self.logits_list.append(calibrated_logits)

#                 if len(self.logits_list) == 1:
#                     next_token_logits = self.logits_list[0]
#                 else:
#                     # 使用torch.stack和torch.mean计算self.logits_list的平均值
#                     logits_tensor = torch.stack(self.logits_list)
#                     next_token_logits = torch.mean(logits_tensor, dim=0)
                
#                 # 截断
#                 # # 定义 beta
#                 # beta = 0.99
#                 # # 获取 next_token_logits 的最大值
#                 # max_logits, _ = torch.max(next_token_logits_base, dim=-1, keepdim=True)
#                 # # 计算阈值
#                 # threshold = beta * max_logits
#                 # # 创建一个全为 -inf 的掩码
#                 # mask = torch.full_like(next_token_logits_base, fill_value=-float('inf'))
#                 # # 获取满足阈值条件的索引
#                 # valid_indices = next_token_logits >= threshold
#                 # # 将满足条件的 logits 保留，不满足条件的设为 -inf
#                 # next_token_logits = torch.where(valid_indices, next_token_logits, mask)

#                 top_k = 100
#                 _, topk_indices_ = torch.topk(next_token_logits_base, top_k, dim=-1)                # 创建一个全为 -inf 的掩码，形状与 next_token_logits 相同
#                 mask = torch.full_like(next_token_logits, fill_value=-float('inf'))
#                 # 使用 scatter 将 next_token_logits 中 top-k 索引对应的位置替换为原始值
#                 next_token_logits = mask.scatter(dim=-1, index=topk_indices_, src=next_token_logits.gather(dim=-1, index=topk_indices_))


#                 # # 校准 logits
#                 # self.logits_list = []
#                 # chunks_num = len(self.w_list)
#                 # # chunks_num = 2
#                 # cali_top_k = chunks_num
#                 # topk_values, self.topk_indices = torch.topk(torch.tensor(self.w_list), cali_top_k)
#                 # imb_type = 'exp'  # 可选'exp', 'step', 'trunk'
#                 # imb_factor = 0.01  # 可调整imb_factor值以改变分布的陡峭程度
#                 # # 生成长尾分布列表
#                 # self.long_tail_weights = generate_long_tail_list(chunks_num, imb_type, imb_factor)
#                 # # topk_indices: 降序序号；long_tail_weights:降序权重
#                 # for i in self.topk_indices:
#                 #     # 只对选出来的topk元素进行计算
#                 #     calibrated_logits = (1 + 1.5) * next_token_logits_base - 1.5 * self.logits_chunks[i]
#                 #     self.logits_list.append(calibrated_logits)
#                 # # 使用long_tail_weights对logits_list进行加权计算
#                 # weighted_logits_sum = sum(logit * weight for logit, weight in zip(self.logits_list[0:cali_top_k], self.long_tail_weights[0:cali_top_k]))
#                 # next_token_logits = weighted_logits_sum / sum(self.long_tail_weights)

#             else:
#                 # prepare model inputs
#                 model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

#                 with torch.no_grad():
#                 # forward pass to get next token
#                     outputs = self(
#                         **model_inputs,
#                         return_dict=True,
#                         output_attentions=output_attentions,
#                         output_hidden_states=output_hidden_states,
#                     )

#                 if synced_gpus and this_peer_finished:
#                     continue  # don't waste resources running the code we don't need
                
#                 # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
#                 # (the clone itself is always small)
#                 next_token_logits = outputs.logits[:, -1, :].clone()##

#             # pre-process distribution
#             next_token_scores = logits_processor(input_ids, next_token_logits)
#             if do_sample:
#                 next_token_scores = logits_warper(input_ids, next_token_scores)

#             # Store scores, attentions and hidden_states when required
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += (next_token_scores,)
#                 if output_logits:
#                     raw_logits += (next_token_logits,)
#                 if output_attentions:
#                     decoder_attentions += (
#                         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
#                     )
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += (outputs.cross_attentions,)

#                 if output_hidden_states:
#                     decoder_hidden_states += (
#                         (outputs.decoder_hidden_states,)
#                         if self.config.is_encoder_decoder
#                         else (outputs.hidden_states,)
#                     )

#             # token selection
#             if do_sample:
#                 probs = nn.functional.softmax(next_token_scores, dim=-1)
#                 next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

#             else:
#                 next_tokens = torch.argmax(next_token_scores, dim=-1)

#             # finished sentences should have their next token be a padding token
#             if has_eos_stopping_criteria:
#                 print(next_tokens)
#                 print(unfinished_sequences)
#                 print(pad_token_id)
#                 print(unfinished_sequences)
#                 next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

#             # update generated ids, model inputs, and length for next step
#             input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
#             if streamer is not None:
#                 streamer.put(next_tokens.cpu())
#             model_kwargs = self._update_model_kwargs_for_generation(
#                 outputs,
#                 model_kwargs,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#             )

#             unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
#             this_peer_finished = unfinished_sequences.max() == 0

#             # This is needed to properly delete outputs.logits which may be very large for first iteration
#             # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
#             del outputs

#         if streamer is not None:
#             streamer.end()

#         if return_dict_in_generate:
#             if self.config.is_encoder_decoder:
#                 return GenerateEncoderDecoderOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     logits=raw_logits,
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions,
#                     cross_attentions=cross_attentions,
#                     decoder_hidden_states=decoder_hidden_states,
#                     past_key_values=model_kwargs.get("past_key_values"),
#                 )
#             else:
#                 return GenerateDecoderOnlyOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     logits=raw_logits,
#                     attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states,
#                     past_key_values=model_kwargs.get("past_key_values"),
#                 )
#         else:
#             return input_ids

def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        calibration_enable = True
        prefilling_stage = True
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if calibration_enable:
                                # prepare model
                # if not first_iteration:
                if prefilling_stage:
                    if input_ids.shape[1] == 1:
                        raise ValueError("input_ids 只有一个元素，无法执行操作。")
                    input_ids_next = input_ids[:, -1]
                    input_ids = input_ids[:, :-1]
                    model_kwargs['attention_mask'] = model_kwargs['attention_mask'][:, :-1]  
                    model_kwargs['cache_position'] = model_kwargs['cache_position'][:-1] 
                    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    
                    # model_inputs['input_ids'] = model_inputs['input_ids'][:, :-1]  
                    # model_inputs['position_ids'] = model_inputs['position_ids'][:, :-1]
                    # model_inputs['cache_position'] = model_inputs['cache_position'][:-1]
                    # model_inputs['attention_mask'] = model_inputs['attention_mask'][:, :-1]  
                    outputs = self(
                                    **model_inputs,
                                    return_dict=True,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                    )
                    input_ids = torch.cat([input_ids, input_ids_next[:, None]], dim=-1)
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs,
                        model_kwargs,
                        is_encoder_decoder=self.config.is_encoder_decoder,
                    )
                    prefilling_stage = False
                
                self.logits_chunks = []

                # permu_num = [1, 10, 100, 1000, 10000, 100000]
                permu_num = [100] * 1
                # 2. 处理每个块
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                for num in permu_num:
                    # 在position_ids上添加扰动
                    # model_inputs['position_ids'] += torch.randint(num, 10 * num, model_inputs['position_ids'].shape, device=model_inputs['position_ids'].device)
                    # model_inputs['position_ids'].clamp_(min=0)
                    # model_inputs['position_ids'].fill_(1)   # 1048576
                    if synced_gpus and this_peer_finished: 
                        continue  # don't waste resources running the code we don't need
                    with torch.no_grad():
                        outputs_chunk = self(
                            **model_inputs,
                            return_dict=True,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            update_past_key_values=True
                        )
                    # 裁剪kv cache
                    model_inputs['past_key_values'].crop(model_inputs['attention_mask'].shape[1]-1)
                    if synced_gpus and this_peer_finished:
                        continue  # don't waste resources running the code we don't need

                    # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                    # (the clone itself is always small)
                    next_token_logits = outputs_chunk.logits[:, -1, :].clone()
                    self.logits_chunks.append(next_token_logits)
                    del outputs_chunk
                    
                # 3. formal next token predict
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                model_inputs["attention_mask"][0].fill_(1)
                with torch.no_grad():
                    outputs = self(
                        **model_inputs,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits_base = outputs.logits[:, -1, :].clone()                

                self.logits_list = []

                for i in range(len(self.logits_chunks)):
                    # 只对选出来的topk元素进行计算
                    # next_token_logits_base_softmax = nn.functional.softmax(next_token_logits_base, dim=-1)
                    # self.logits_chunks[i] = nn.functional.softmax(self.logits_chunks[i], dim=-1)
                    calibrated_logits = (1 + 1.5) * next_token_logits_base - 1.5 * self.logits_chunks[i]
                    # calibrated_logits = next_token_logits_base - 0.1 * self.logits_chunks[i]
                    # calibrated_logits = next_token_logits_base_softmax - self.logits_chunks[i]
                    self.logits_list.append(calibrated_logits)

                if len(self.logits_list) == 1:
                    next_token_logits = self.logits_list[0]
                else:
                    # 使用torch.stack和torch.mean计算self.logits_list的平均值
                    logits_tensor = torch.stack(self.logits_list)
                    next_token_logits = torch.mean(logits_tensor, dim=0)
                
                top_k = 100
                _, topk_indices_ = torch.topk(next_token_logits_base, top_k, dim=-1)                # 创建一个全为 -inf 的掩码，形状与 next_token_logits 相同
                mask = torch.full_like(next_token_logits, fill_value=-float('inf'))
                # 使用 scatter 将 next_token_logits 中 top-k 索引对应的位置替换为原始值
                next_token_logits = mask.scatter(dim=-1, index=topk_indices_, src=next_token_logits.gather(dim=-1, index=topk_indices_))

            else:
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # forward pass to get next token
                outputs = self(**model_inputs, return_dict=True)

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].clone()
            
            ##########################################################################################
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        
def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        logits_warper: Optional[LogitsProcessorList] = None,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "bloom",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(
                        **inputs_per_sub_batch,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)

            else:  # Unchanged original behavior
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            if do_sample:
                next_token_scores_processed = logits_warper(input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]
        
def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
       
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        use_dynamic_cache_by_default = False
        if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs["past_key_values"] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    generation_config.max_length,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                if cache_config.backend == "quanto" and not is_quanto_available():
                    raise ImportError(
                        "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                        "Please install it via  with `pip install quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )

                model_kwargs["past_key_values"] = cache_class(cache_config)
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get("past_key_values", None)
            if past is None:
                model_kwargs["past_key_values"] = DynamicCache()
                use_dynamic_cache_by_default = True
            elif isinstance(past, tuple):
                model_kwargs["past_key_values"] = DynamicCache.from_legacy_cache(past)
                use_dynamic_cache_by_default = True

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation == "static":
                raise ValueError("assisted generate is not supported with `static_cache`")
            if self._is_stateful:
                # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                )

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
            )

            # 12. prepare logits warper (if `do_sample` is `True`)
            prepared_logits_warper = (
                self._get_logits_warper(
                    generation_config,
                    device=input_ids.device,
                )
                if generation_config.do_sample
                else None
            )

            # 13. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(
                    f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
                )

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache if needed
        if use_dynamic_cache_by_default and generation_config.return_legacy_cache:
            if isinstance(result, ModelOutput) and hasattr(result, "past_key_values"):
                if isinstance(result.past_key_values, DynamicCache):
                    result.past_key_values = result.past_key_values.to_legacy_cache()
        return result



def generate_replace():
    # print("Nothing happend.")
    transformers.generation.utils.GenerationMixin._sample = _sample
    # transformers.generation.utils.GenerationMixin._beam_search = _beam_search
    # transformers.generation.utils.GenerationMixin.generate = generate