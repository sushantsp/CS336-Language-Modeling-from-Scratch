## Problem (unicode2): Unicode Encodings

a. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.

Answer : UTF-16 and UTF-32 are sparser representation of characters as compared to UTF-8. Meaning to represent the same character ohter two end up with more bytes which are always 'null'. This increases the embedding length downstream. Hence reduced context length. or reduced content in a context


b. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results 

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

Answer : 
 bytes can only contain ASCII literal characters. Characters other that ASCII cannot be processed by this code specially `bytes([b])`. `b` is expected to be ASCII character. 
 `decode_utf8_bytes_to_str_wrong(b'hELLP こ')` will fail because of the japanese character which is not ASCII. 

c. Give a two byte sequence that does not decode to any Unicode character(s).

Answer : `b'\xe3\x81'` this does not get decoded into any unicode character or string. because there is no corresponding representation of this in Unicode. `b'\xe3\x81\x93'` represents the korean character `こ` but on just removing the last byte from the sequence makes it unrepresentable.