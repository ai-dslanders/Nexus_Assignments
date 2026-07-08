sample = "to be or not to be that is the question"
# 1. Build a set `unique_words` containing every distinct word.
# 2. Build a dict `word_counts` mapping each word to the number of times it appears.
#    (Hint: .split() + a simple loop)
# 3. Print the two structures and explain (in a comment) their main difference.
# TODO: your code here
uw1 = []
uw1 = sample.split()
uw2 = list(set(uw1))
uw2.sort()
repeats=[]
for element in uw2:
    count=uw1.count(element)
    repeats.append(count)
uwd = {}
for i in range(len(repeats)):
    uwd[uw2[i]] = repeats[i]
print(uw1)
print(uw2)
print(repeats)
print(uwd)