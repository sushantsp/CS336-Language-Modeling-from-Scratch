### What is `Counter`?

* `Counter` is a specialized dictionary from `collections`
* Designed **specifically for counting**
* Replaces manual “dictionary + initialization + increment” patterns

```python
from collections import Counter
```

---

### Why use `Counter` instead of `dict`?

* No manual initialization
* Cleaner syntax
* Built-in counting utilities
* Supports arithmetic operations

---

### Basic Counting (stream / API-like)

**Dictionary approach (tedious):**

```python
d = {}
for obj in object_list:
    d[obj] = 0

received = random.choice(object_list)
d[received] += 1
```

**Counter approach (clean):**

```python
c = Counter()
received = random.choice(object_list)
c[received] += 1
```

✅ No `KeyError`, no pre-initialization.

---

### Counting from a List (batch data)

```python
occurrences = random.choices(object_list, k=100)
c = Counter(occurrences)
```

Equivalent dictionary version requires:

* initialization loop
* counting loop

---

### Counting Words / Tokens (very common)

```python
sentence = "this is a test this is"
Counter(sentence.split())
```

Useful in:

* NLP
* log analysis
* token frequency tasks

---

### `most_common()` (major advantage)

```python
c.most_common()
c.most_common(2)
c.most_common(1)[0]
```

Returns:

```python
[('a', 21), ('c', 20), ...]
```

❌ Dictionary requires manual max-tracking logic
✅ Counter does it in one call

---

### `total()` (Python ≥ 3.10)

```python
c.total()
```

Returns sum of all counts
(Not available in Python 3.9 and earlier)

---

### Counter Arithmetic (not possible with dicts)

```python
c1 = Counter(a=3, b=5, c=4)
c2 = Counter(a=1, b=2, c=1)

c1 - c2
```

Result:

```python
Counter({'b': 3, 'c': 3, 'a': 2})
```

❌ `dict - dict` → TypeError
✅ `Counter - Counter` → element-wise subtraction

---

### Creating Counters Manually

```python
c = Counter(a=20, b=15, c=12)
```

---

### `elements()` — expand counts

```python
list(c.elements())
```

Example:

```python
Counter(a=2, b=1) → ['a', 'a', 'b']
```

* Returns an **iterator**
* Convert to `list()` to view

---

### When NOT to use Counter

* When dictionary values are **not counts**
* When key–value mapping has semantic meaning beyond frequency

---

### Mental Model

> **If your dictionary’s only job is counting → use `Counter`.**

---
