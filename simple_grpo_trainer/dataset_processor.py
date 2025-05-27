from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def extract_answer(example: dict[str, str]):
    """
    Extract the answer from the example.

    Args:
        example (dict[str, str]): The example to extract the answer from.

    Returns:
        dict[str, str]: The example with the answer extracted.
    """
    answer_loc = example["answer"].find("### ")
    example["answer"] = example["answer"][answer_loc + 4:]
    return example


def get_gsm8k_dataset():
    """
    Load the GSM8K dataset.

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset.map(extract_answer)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    return train_dataset, test_dataset

def tokenize_example(example: dict[str, str], tokenizer: AutoTokenizer):
    """
    Tokenize the dataset.

    Args:
        example (dict[str, str]): The example to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use.

    Returns:
        Dataset: The tokenized dataset.
    """
    system_prompt= """
    You are a helpful assistant that will use reasoning, long chain of thought, backtracking, and 
    self-reflection to answer math problems.
    """
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": example["question"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    example["prompt"] = prompt
    return example

def create_dataloader(
    dataset: Dataset, 
    is_train: bool = False,
    batch_size: int = 1,
    ):
    """
    Create a dataloader for the dataset.

    Args:
        dataset (Dataset): The dataset to create a dataloader for.
        tokenizer (AutoTokenizer): The tokenizer to use.
        is_train (bool): Whether the dataset is for training.

    Returns:
        DataLoader: The dataloader.
    """
    do_shuffle = False
    if is_train:
        do_shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle)
    return dataloader

if __name__ == "__main__":
    train_dataset, test_dataset = get_gsm8k_dataset()
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    train_dataset = train_dataset.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer})
    test_dataset = test_dataset.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer})
    train_dataloader = create_dataloader(train_dataset, is_train=False, batch_size=2)
    test_dataloader = create_dataloader(test_dataset, is_train=False, batch_size=2)
    for batch in train_dataloader:
        print(batch)
        break