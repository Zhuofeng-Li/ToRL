# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import os
import torch
import torch.distributed
from tensordict import TensorDict
import requests
from multiprocessing import Pool
from functools import partial
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from qwen_agent.tools.python_executor import PythonExecutor
from qwen_agent.tools.code_interpreter import CodeInterpreter
from qwen_agent.utils.utils import print_traceback
from typing import Tuple
import json5
import pdb
import json
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


OBS_START = '```output'
OBS_END = '\n```\n'
def extract_program(result: str, last_only=True):
    """
    extract the program after "```python", and before "```"
    """
    program = ''
    start = False
    for line in result.split('\n'):
        if line.startswith('```python') or line.endswith('```python'):
            if last_only:
                program = ''  # only extract the last program
            else:
                program += '\n# ========\n'
            start = True
        elif line.startswith('```'):
            start = False
        elif start:
            program += line + '\n'
    if start:
        # the code is incomplete
        program = ''
    return program

def _detect_tool(text: str) -> Tuple[bool, str, str, str]:
    program = extract_program(text)
    if program:
        program = json.dumps({'code': program}, ensure_ascii=False)
    return (program != ''), PythonExecutor.name, program, text

def send_request(json_data, url):
    try:
        # url = 'http://0.0.0.0:8080/run_code'
        response = requests.post(url, json=json_data, timeout=10)
        return response.json()  # 返回响应的 JSON 数据
    except:
        return {"error": "unknown"}


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.executor=PythonExecutor()
        # self.code_interpreter=CodeInterpreter()
    
    def _get_prompts_and_indices(self, samples_info):
        prompts, indices=[], []
        for index, info in enumerate(samples_info):
            if not info['stop']:
                prompts.append(info['sequence'])
                indices.append(info['index'])
        return prompts, indices

    # def code_interpreter_batch_call(self, tool_inputs):
    #     with Pool(processes=min(len(tool_inputs),os.cpu_count(), 32)) as pool:
    #         results = pool.map(self.code_interpreter.call, tool_inputs)
    #     def postproc(result):
    #         report=result.split("```")[0].strip()
    #         output=result.split("```")[-1].split("```")[-1].strip()
    #         if report=="stdout:": report="Done"
    #         return (output, report)
    #     results=[postproc(result) for result in results]
    #     return results

    def code_interpreter_batch_call(self, tool_inputs, timeout=20):
        tool_inputs=[{'code': tool_input,'language': 'python'} for tool_input in tool_inputs]
        results = [None] * len(tool_inputs) 
        with ThreadPoolExecutor(max_workers=max(min(len(tool_inputs), os.cpu_count(), 64), 1)) as executor:
            future_to_index = {executor.submit(send_request, input, self.config.url): i for i, input in enumerate(tool_inputs)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=timeout)
                    results[index] = result
                except:
                    results[index] = {"run_result": {"stdout": "Error", "stderr": "TimeoutError"}}
        
        def postproc(output):
            try:
                if str(output['run_result']['return_code'])=='0' or len(str(output['run_result']['stdout'])) != 0:
                    return output['run_result']['stdout'], "Done"
                else:
                    return output['run_result']['stdout'], output['run_result']['stderr'].strip()
            except Exception: 
                return "Error", "UnknownError"
        results=[postproc(result) for result in results]
        return results

    def _tokenize_and_find_mask_token_indices(self, sample_info):
        response=sample_info['response']
        mask_str_ranges=sample_info['mask_info']

        encoding=self.tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        
        response_token_ids=encoding['input_ids']

        offset_mapping_tensor=torch.tensor(encoding['offset_mapping'], dtype=torch.long)
        token_starts = offset_mapping_tensor[:,0]
        token_ends = offset_mapping_tensor[:,1]

        mask_tensor=torch.ones(len(response_token_ids))
        for mask_str_range in mask_str_ranges:
            start_index, end_index=mask_str_range[0], mask_str_range[1]
            mask = (token_starts < end_index) & (token_ends > start_index) & (token_starts >= start_index)
            mask_tensor[mask]=0 

        return response_token_ids, mask_tensor


    def _tir_generate(self, prompts=None, sampling_params=None, prompt_token_ids=None, use_tqdm=False):
        sampling_params=copy.deepcopy(sampling_params)
        # prompts=self.tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=True)
        prompts=[self.tokenizer.decode(prompt['prompt_token_ids'], skip_special_tokens=False) for prompt in prompts]
        prompts=[prompt for prompt in prompts for _ in range(sampling_params.n) ]
        sampling_params.n=1
        sampling_params.detokenize=True
        sampling_params.stop=["```output"]
        samples_info=[{"prompt": prompt, "sequence": prompt, "response": "", "stop": False, "finish_reason": None,"index": index, "mask_info": [], "execution_pass": 0} for index, prompt in enumerate(prompts)]
        program2output=[]
        num_llm_calls_available=copy.deepcopy(self.config.num_llm_calls_available)
        while num_llm_calls_available >= 0:
            if num_llm_calls_available==0: sampling_params.stop=None
            num_llm_calls_available-=1
            # llm generate response, stop at eos token or ```output
            input_prompts, indices=self._get_prompts_and_indices(samples_info)
            input_prompts = [{
                'prompt_token_ids': self.tokenizer.encode(x, add_special_tokens=False)[:self.config.prompt_length+self.config.response_length]} for x in input_prompts]
            outputs = self.inference_engine.generate(prompts=input_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
            sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
            responses=[x.outputs[0].text for x in sorted_outputs]
            finish_reason=[x.outputs[0].finish_reason for x in sorted_outputs]
            stop_reason=[x.outputs[0].stop_reason for x in sorted_outputs]
            if num_llm_calls_available==-1:
                for i ,index in enumerate(indices):
                    samples_info[index]['response']+=responses[i]
                    samples_info[index]['sequence']+=responses[i]
                    samples_info[index]['stop']=True
                    samples_info[index]['finish_reason']=finish_reason[i]
                break

            def _python_execution(finish_reason, stop_reason):
                if finish_reason=='stop' and stop_reason==None: return False
                if finish_reason=='stop' and stop_reason=='```output': return True
                if finish_reason=='length': False
                return False
            is_execution=[_python_execution(finish_reason[i], stop_reason[i]) for i in range(len(finish_reason))]
            # check if all samples are finished
            if all([not x for x in is_execution]): break

            # prepare for python execution
            tool_infos=[ _detect_tool(response) for response in responses]
            tool_indices=[]
            tool_inputs=[]
            for i, tool_info in enumerate(tool_infos):
                if tool_info[0] and is_execution[i]:
                    tool_indices.append(i)
                    tool_inputs.append(tool_info[2])
            
            def postproc_observation(observation):
                execution_pass=0
                try:
                    observation_list=observation
                    if observation_list[-1] == 'Done':
                        observation = observation_list[0]
                        execution_pass=1
                    else:
                        observation = observation_list[-1]
                except Exception:
                    observation="Error"
                if "Error" in observation: observation=observation.strip().split("\n")[-1]
                if len(observation.strip())==0: observation="timeout_decorator.timeout_decorator.TimeoutError: 'Timed Out'"
                observation = observation.strip()
                if len(observation)>=256:
                    observation = observation[:128]+"..."+observation[-128:]
                observation = f'{OBS_START}\n{observation}{OBS_END}'
                return observation, execution_pass

            # execute python code

            # observations=self.executor.batch_apply([json5.loads(x)['code'] for x in tool_inputs])
            observations=self.code_interpreter_batch_call([json5.loads(x)['code'] for x in tool_inputs])
            
            # construction responses from observations
            responses=[response+"\n" if not response.endswith('\n') else response for response in responses]
            responses_w_res=copy.deepcopy(responses)
            execution_passes=[0 for _ in range(len(responses))]
            for i, index in enumerate(tool_indices):
                processed_observation=postproc_observation(observations[i])
                responses_w_res[index]+=processed_observation[0]
                execution_passes[index]=processed_observation[1]
            
            # program2output.append([{"code": tool_input, "answer": postproc_observation(observations[idx])} for idx, tool_input in enumerate(tool_inputs)])
            # update samples_info
            for i ,index in enumerate(indices):
                mask=[ len(responses[i]) + len('```output'), len(responses_w_res[i]) ]
                samples_info[index]['mask_info'].append(mask)
                samples_info[index]['response']+=responses_w_res[i]
                samples_info[index]['sequence']+=responses_w_res[i]
                samples_info[index]['stop']=not is_execution[i]
                samples_info[index]['finish_reason']=finish_reason[i]
                samples_info[index]['execution_pass']=execution_passes[i]
        
        for i, line in enumerate(samples_info):
            if samples_info[i]['finish_reason']!='length': samples_info[i]['response']+=self.tokenizer.eos_token
        
        responses_ids=[]
        tool_output_masks=[]
        execution_passes=[]
        for idx, sample_info in enumerate(samples_info):
            response_id, tool_output_mask = self._tokenize_and_find_mask_token_indices(sample_info)
            responses_ids.append(response_id[:self.config.response_length])
            tool_output_masks.append(tool_output_mask[:self.config.response_length])
            execution_passes.append(sample_info['execution_pass'])
            # save id and mask to check correctness
        #     samples_info[idx]['responses_id']=response_id[:self.config.response_length]
        #     samples_info[idx]['tool_output_mask']=tool_output_mask[:self.config.response_length].tolist()
        

        # with open("/mnt/bn/seedllm3-lixuefeng-2/code/o1/verl-tir/sample_infos.json", 'w', encoding='utf-8') as f:
        #     json.dump(samples_info, f, ensure_ascii=False, indent=2)
        return responses_ids, tool_output_masks, torch.tensor(execution_passes, dtype=torch.long)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            response, tool_output_masks, execution_passes = self._tir_generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        # response = []
        # for output in outputs:
            # for sample_id in range(len(output.outputs)):
                # response.append(output.outputs[sample_id].token_ids)
        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)
        tool_output_masks = pad_2d_list_to_length(tool_output_masks, 1,
                                         max_length=self.config.response_length).to(idx.device).int()
        execution_passes = execution_passes.to(idx.device).int()
        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
            if 'multi_modal_inputs' in non_tensor_batch.keys():
                non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                            self.config.n)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        # response_attention_mask = response_attention_mask & tool_output_masks
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'tool_output_masks': tool_output_masks,
                'position_ids': position_ids,
                'execution_passes': execution_passes,            
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
