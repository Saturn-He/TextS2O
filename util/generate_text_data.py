import argparse

from datasets import PairedImageDirDataset
from text_encoder import SYSTEM_PROMPT_EN, ensure_text_data


def get_args_parser():
    parser = argparse.ArgumentParser("Generate text data from SAR/OPT pairs")
    parser.add_argument("--sar_path", required=True, help="Path to SAR image directory")
    parser.add_argument("--opt_path", required=True, help="Path to optical image directory")
    parser.add_argument("--output_dir", required=True, help="Directory to write text files")
    parser.add_argument("--llm_model_name", default="Qwen2-VL-72B", help="LLM model name for text generation")
    parser.add_argument("--qwen_prompt", default=SYSTEM_PROMPT_EN, help="Prompt for the Qwen VL API")
    return parser


def main(args):
    dataset = PairedImageDirDataset(args.sar_path, args.opt_path, transform=None)
    ensure_text_data(
        dataset.sar_files,
        dataset.opt_files,
        args.output_dir,
        args.llm_model_name,
        args.qwen_prompt,
        transform=None,
    )


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
