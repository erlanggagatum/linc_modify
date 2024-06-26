import os
import fnmatch
import json
import pathlib
from warnings import warn

import torch
import openai
import datasets
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from eval.args import RunnerArguments, HFArguments, OAIArguments, GenerationArguments
from eval.evaluator import HFEvaluator, OAIEvaluator
from eval.tasks import ALL_TASKS

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()




def main():
    # def main():
    args = HfArgumentParser(
        [RunnerArguments, HFArguments, OAIArguments, GenerationArguments]
    ).parse_known_args()[0]

    __file__ = ''
    model_name = 'bigcode/starcoderplus'
    base = 'folio'
    batch_size = 5
    max_length=8192 # max model context including prompt
    shot = '2'
    mode = 'neurosymbolic'
    task = f'{base}-{mode}-{shot}shot'
    # run_id='${model#*/}_${task}'
    run_id = f"{model_name}_{task}"

    # print('$')
    args.top_k = 2
    args.output_dir = pathlib.Path('../output/')
    args.model = model_name
    args.temperature = 0.8
    args.max_length_generation = 1024
    # args.generation_only = True
    args.save_generations_raw_path = args.output_dir / f'{run_id}_generations_raw.json'
    args.save_generations_prc_path = args.output_dir / f'{run_id}_generations_prc.json'
    args.save_references_path = args.output_dir / f'{run_id}_references.json'
    args.save_results_path = args.output_dir / f'{run_id}_results.json'
    args.save_generations_raw_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_generations_prc_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_references_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_results_path.parent.mkdir(parents=True, exist_ok=True)
    args.allow_code_execution = True
    args.tasks = task
    args.precision = 'fp32'

    # print(save_generations_raw_path)

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = set()
        for pattern in args.tasks.split(","):
            for matching in fnmatch.filter(ALL_TASKS, pattern):
                task_names.add(matching)
        task_names = list(task_names)
    
    # accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    results = {}
    if args.generations_path:
        if accelerator.is_main_process:
            print("Evaluation only mode")
        evaluator = HFEvaluator(accelerator, None, None, args)
        for task in task_names:
            results[task] = evaluator.evaluate(task)
    else:
        evaluator = None
        
        if evaluator is None:
            dict_precisions = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            if args.precision not in dict_precisions:
                raise ValueError(
                    f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
                )
            print(f"Loading the model and tokenizer from HF (in {args.precision})")
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    revision=args.revision,
                    torch_dtype=dict_precisions[args.precision],
                    trust_remote_code=args.trust_remote_code,
                    use_auth_token=args.use_auth_token,
                )
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model,
                    revision=args.revision,
                    use_auth_token=args.use_auth_token,
                    truncation_side="left",
                )
            if not tokenizer.eos_token:
                if tokenizer.bos_token:
                    tokenizer.eos_token = tokenizer.bos_token
                    print("bos_token used as eos_token")
                else:
                    raise ValueError("No eos_token or bos_token found")
            tokenizer.pad_token = tokenizer.eos_token
            evaluator = HFEvaluator(accelerator, model, tokenizer, args)
            print('Finish loading model from HF')
            
        for task in task_names:
            if args.generation_only:
                print('generation only ---')
                if accelerator.is_main_process:
                    print("Generation mode only")
                generations_prc, generations_raw, references = evaluator.generate_text(
                    task
                )
                if accelerator.is_main_process:
                    if args.save_generations_raw:
                        with open(args.save_generations_raw_path, "w") as fp:
                            json.dump(generations_raw, fp)
                            print("raw generations were saved")
                    if args.save_generations_prc:
                        with open(args.save_generations_prc_path, "w") as fp:
                            json.dump(generations_prc, fp)
                            print("processed generations were saved")
                    if args.save_references:
                        with open(args.save_references_path, "w") as fp:
                            json.dump(references, fp)
                            print("references were saved")
            else:
                print('evaluation only ---')
                
                results[task] = evaluator.evaluate(task)
                
        results["config"] = {"model": args.model}
        if not args.generation_only:
            dumped = json.dumps(results, indent=2, sort_keys=True)
            if accelerator.is_main_process:
                print(dumped)

            if args.save_results:
                with open(args.save_results_path, "w") as f:
                    f.write(dumped)
                    
if __name__ == "__main__":
    main()
