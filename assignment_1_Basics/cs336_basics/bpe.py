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

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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
def pretokenization(text : str, special_tokens: list[str]) -> list[list[bytes]]:

    # # Remove special tokens from the text
    # for token in special_tokens:
    #     text = text.replace(token, "")    

    # pre tokenization
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # # Get a list of pre tokens using the pattern above
    # pre_tokens_iterator = re.finditer(PAT, text)
    # pre_tokens = [match.group().encode('utf-8') for match in pre_tokens_iterator]

    # logger.info(f"Pretokens created - {pre_tokens[0]}")
    # # break down each pre token into one byte chunk - b'H'
    # pre_tokens_bytes = []
    # for token in pre_tokens:
    #     # pre_tokens_bytes.append([bytes([id]) for id in token])
    #     pre_tokens_bytes.append(list(token))
    # # logger.info("single byte object list created for each word byte")

    # ==============
    # for token in special_tokens:
    #     text = text.replace(token, " ")

    # pre_tokens_bytes = []
    # for match in PAT.finditer(text):
    #     token = match.group().encode("utf-8")

    #     # Each element MUST be bytes
    #     pre_tokens_bytes.append(
    #         [token[i:i+1] for i in range(len(token))]
    #     )
    # ===============

    pre_tokens_bytes = []
    chunks = [text]
    for token in special_tokens:
        new_chunks = []
        for chunk in chunks:
            new_chunks.extend(chunk.split(token))
        chunks = new_chunks

    for chunk in chunks:
        for match in PAT.finditer(chunk):
            token = match.group().encode("utf-8")
            pre_tokens_bytes.append(
                [token[i:i+1] for i in range(len(token))]
            )
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


def pretokenize_parallel(text_corpus: str, special_token: str, num_workers: int =1) -> list[list[bytes]]:

    text_chunks = text_corpus.split(special_token)
    all_tokens = []

    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        for chunk_token in exe.map(pretokenization_chunk, text_chunks):
            all_tokens.extend(chunk_token)

    return all_tokens


def get_pair_freq_counts(pre_tokens_bytes: Counter) -> dict[tuple[bytes], int]:
    # Get a freq count of consecutive pair using each pre token - we maintain the pretoken boundaries
    freq_count_bp = Counter()


    for token, freq in pre_tokens_bytes.items():
        for a,b in zip(token[:], token[1:]):
            freq_count_bp[(a,b)] += freq


    # logger.info("Frequency count of each pair is done.")
    return freq_count_bp


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


# def train_bpe(text: str, special_tokens:list[str], vocab_size:int, num_workers: int = 1):

#     # # Remove special tokens from the text
#     # for token in special_tokens:
#     #     text = text.replace(token, "")

#     vocab = init_vocab(special_tokens)
#     # initial merges - ordered from earliest-created to latest
#     merges = []
#     initial_vocab_size = len(vocab.keys())

#     no_of_merges = vocab_size - initial_vocab_size
#     # logger.info(f"No of merges that are required to achieve vocab size {vocab_size} is {no_of_merges}")

#     # tokens_list  = pretokenize_parallel(text, special_token='<|endoftext|>')
#     if num_workers > 1:
#         tokens_list = pretokenize_parallel(text, '<|endoftext|>', num_workers)
#     else:
#         tokens_list = pretokenization(text, special_tokens)

#     tokens = Counter(tuple(tok) for tok in tokens_list)

#     # logger.info('pretokenization completed')

#     # logger.info("Merging stared ...")
#     newtoken_id = max(vocab.keys()) + 1
#     freq_pair_cnt = get_pair_freq_counts(tokens)
#     for i in range(no_of_merges):

#         # Add check for empty frequency counts
#         if not freq_pair_cnt:
#             # logger.warning("No more pairs to merge - stopping early")
#             break
        
#         try:
#             # Get the top pair - making sure we have lexicographical order taken into consideration
#             top_pair, count = max(freq_pair_cnt.items(), key = lambda x: (x[1], x[0])) # evaulate by count and pair. first count then pair to break a tie in case of 

#         except ValueError as e:
#             # logger.error(f"Error finding max pair: {e}")
#             break

#         # logger.info(f"top pair selected is {top_pair}")
#         # update tokens
#         tokens = merge(tokens, top_pair)

#         # add that pair into new token and update the vocab 
#         vocab[newtoken_id] = top_pair[0] + top_pair[1]
#         newtoken_id += 1

#         # logger.info(f"new vocab token {vocab[newtoken_id]} with {newtoken_id}")

#         # keep a track of merges - ordered from earliest-created to latest
#         merges.append(top_pair)

#         freq_pair_cnt = get_pair_freq_counts(tokens)
#     # logger.info("Merging completed")

#     # print(vocab, "=="*10, merges)
#     return vocab, merges



def train_bpe(text: str, vocab_size:int, special_tokens:list[str],  num_workers: int = 1):

    # # Remove special tokens from the text
    # for token in special_tokens:
    #     text = text.replace(token, "")

    vocab = init_vocab(special_tokens)
    # initial merges - ordered from earliest-created to latest
    merges = []


    # logger.info(f"No of merges that are required to achieve vocab size {vocab_size} is {no_of_merges}")

    # Pretokenization
    if num_workers > 1:
        tokens_list = pretokenize_parallel(text, '<|endoftext|>', num_workers)
    else:
        tokens_list = pretokenization(text, special_tokens)

    tokens = Counter(tuple(tok) for tok in tokens_list)

    pair_freqs = Counter()
    for token, freq in tokens.items():
        for a, b in zip(token, token[1:]):
            pair_freqs[(a, b)] += freq


    # logger.info('pretokenization completed')

    # logger.info("Merging stared ...")
    newtoken_id = max(vocab.keys()) + 1

    no_of_merges = vocab_size - len(vocab)
    for i in range(no_of_merges):

        if not pair_freqs:
            break
        
        top_pair = max(pair_freqs.items(), key = lambda x: (x[1], x[0]))[0] # evaulate by count and pair. first count then pair to break a tie in case of 
        # Add pair to the merges list
        merges.append(top_pair)

        # merge
        p0 , p1 = top_pair
        p0p1 = p0 + p1
        # add that pair into new token and update the vocab 
        vocab[newtoken_id] = p0p1
        new_tokens = Counter()

        pair_freqs_local = pair_freqs  # FIX 3: local cache

        # update tokens
        for pretoken_tuple, freq in tokens.items():
            new_token = []
            i = 0

            n = len(pretoken_tuple)
            while i < n:
                if i + 1 < n and (pretoken_tuple[i], pretoken_tuple[i+1]) == top_pair:
                    has_left = len(new_token) > 0
                    has_right = i + 2 < n

                    if has_left:
                        left = new_token[-1]
                        pair =(left, p0p1)
                        pair_freqs_local[pair] += freq

                        old = (left, p0)
                        pair_freqs_local[old] -= freq
                        if pair_freqs_local[old] <= 0:
                            del pair_freqs_local[old] 


                    if has_right:
                        right = pretoken_tuple[i+2]
                        pair = (p0p1, right)
                        pair_freqs_local[pair] += freq

                        old = (p1, right)
                        pair_freqs_local[old] -= freq
                        if pair_freqs_local[old] <= 0:
                            del pair_freqs_local[old]


                    pair_freqs_local[top_pair] -= freq
                    if pair_freqs_local[top_pair] <= 0:
                        del pair_freqs_local[top_pair]

                    new_token.append(p0p1)
                    i+=2

                else:
                    new_token.append(pretoken_tuple[i])
                    i+=1

            new_tokens[tuple(new_token)] += freq

        tokens = new_tokens
        newtoken_id += 1

    return vocab, merges