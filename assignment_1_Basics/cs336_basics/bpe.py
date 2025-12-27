import regex as re
import logging
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

# Set up logging to show INFO and above
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.disabled = True

def init_vocab(special_tokens: list) -> dict[int, bytes]:
    # Initialise the vocab without special tokens 
    vocab = {idx : bytes([idx]) for idx in range(256)}

    for sp_token in special_tokens:
        newtoken_id = max(vocab.keys()) + 1
        # special tokens - <|endoftext|>
        vocab[newtoken_id] = sp_token.encode('utf-8')

    # logger.info("Initial vocab with special token included created.")

    return vocab

# "Helllow owjrld world"
def pretokenization(text : str) -> list[list[bytes]]:

    # pre tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Get a list of pre tokens using the pattern above
    pre_tokens_iterator = re.finditer(PAT, text)
    pre_tokens = [match.group().encode('utf-8') for match in pre_tokens_iterator]

    logger.info(f"Pretokens created - {pre_tokens[0]}")
    # break down each pre token into one byte chunk - b'H'
    pre_tokens_bytes = []
    for token in pre_tokens:
        pre_tokens_bytes.append([bytes([id]) for id in token])
    logger.info("single byte object list created for each word byte")
    return pre_tokens_bytes


def pretokenization_chunk(text_chunk : str) -> list[list[bytes]]:

    # pre tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Get a list of pre tokens using the pattern above
    pre_tokens_iterator = re.finditer(PAT, text_chunk)
    pre_tokens = [match.group().encode('utf-8') for match in pre_tokens_iterator]

 
    # break down each pre token into one byte chunk - b'H'
    pre_tokens_bytes = []
    for token in pre_tokens:
        pre_tokens_bytes.append([bytes([id]) for id in token])

    return pre_tokens_bytes


def pretokenize_parallel(text_corpus: str, special_token: str, num_workers: int =4) -> list[list[bytes]]:

    text_chunks = text_corpus.split(special_token)
    all_tokens = []

    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        for chunk_token in exe.map(pretokenization_chunk, text_chunks):
            all_tokens.extend(chunk_token)

    return all_tokens


def get_pair_freq_counts_old(pre_tokens_bytes: list[list[bytes]]) -> dict[tuple[bytes], int]:
    # Get a freq count of consecutive pair using each pre token - we maintain the pretoken boundaries
    freq_count_bp = {}

    for byte_tokens in pre_tokens_bytes:
        for i in range(len(byte_tokens)):
            if i < len(byte_tokens) - 1:
                freq_count_bp[(byte_tokens[i], byte_tokens[i+1])] = freq_count_bp.get((byte_tokens[i], byte_tokens[i+1]), 0) + 1
    logger.info("Frequency count of each pair is done.")
    return freq_count_bp


def get_pair_freq_counts(pre_tokens_bytes: Counter) -> dict[tuple[bytes], int]:
    # Get a freq count of consecutive pair using each pre token - we maintain the pretoken boundaries
    freq_count_bp = Counter()


    for token, freq in pre_tokens_bytes.items():
        for a,b in zip(token[:], token[1:]):
            freq_count_bp[(a,b)] += freq


    # logger.info("Frequency count of each pair is done.")
    return freq_count_bp



def merge_old(old_tokens, top_pair) -> list[list[bytes]]:
    
    logger.info(f"Merging a {top_pair} into tokens")
    # update pre_token_bytes with merges

    new_tokens = []
    for b_tokens in old_tokens:
        new_list = []
        i = 0
        while i < len(b_tokens):
            if i < len(b_tokens) - 1 and (b_tokens[i], b_tokens[i+1]) == top_pair:
                new_list.append(top_pair[0] + top_pair[1])
                i += 2
            else:
                new_list.append(b_tokens[i])
                i +=1
        new_tokens.append(new_list)
    logger.info(f"Merging of pair - {top_pair} into tokens completed.")
    return new_tokens


def merge(old_tokens: Counter, top_pair) -> Counter:
    
    # update pre_token_bytes with merges

    new_tokens = Counter()

    for token, freq in old_tokens.items():
        merged = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i+1]) == top_pair:
                merged.append(top_pair[0] + top_pair[1])
                i += 2
            else:
                merged.append(token[i])
                i +=1
        new_tokens[tuple(merged)] += freq
    return new_tokens


def train_bpe(text: str, special_tokens:list[str], vocab_size:int, num_workers: int = 4):

    # # Remove special tokens from the text
    # for token in special_tokens:
    #     text = text.replace(token, "")

    vocab = init_vocab(special_tokens)
    # initial merges - ordered from earliest-created to latest
    merges = []
    initial_vocab_size = len(vocab.keys())

    no_of_merges = vocab_size - initial_vocab_size
    # logger.info(f"No of merges that are required to achieve vocab size {vocab_size} is {no_of_merges}")


    tokens_list  = pretokenize_parallel(text, special_token='<|endoftext|>')

    tokens = Counter(tuple(tok) for tok in tokens_list)

    # logger.info('pretokenization completed')

    # logger.info("Merging stared ...")
    newtoken_id = max(vocab.keys()) + 1
    for i in range(no_of_merges):

        freq_pair_cnt = get_pair_freq_counts(tokens)

        # Add check for empty frequency counts
        if not freq_pair_cnt:
            # logger.warning("No more pairs to merge - stopping early")
            break
        
        try:
            # Get the top pair - making sure we have lexicographical order taken into consideration
            top_pair, count = max(freq_pair_cnt.items(), key = lambda x: (x[1], x[0])) # evaulate by count and pair. first count then pair to break a tie in case of 

        except ValueError as e:
            # logger.error(f"Error finding max pair: {e}")
            break

        logger.info(f"top pair selected is {top_pair}")
        # update tokens
        tokens = merge(tokens, top_pair)

        # add that pair into new token and update the vocab 
        vocab[newtoken_id] = top_pair[0] + top_pair[1]
        newtoken_id += 1

        # logger.info(f"new vocab token {vocab[newtoken_id]} with {newtoken_id}")

        # keep a track of merges - ordered from earliest-created to latest
        merges.append(top_pair)
    # logger.info("Merging completed")

    # print(vocab, "=="*10, merges)
    return vocab, merges