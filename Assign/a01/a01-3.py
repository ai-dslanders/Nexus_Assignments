# Start with an empty shopping list (list).
# 1. Append at least 4 items supplied in one line of user input (comma-separated).
# 2. Convert the list to a *tuple* called immutable_basket.
# 3. Print the third item using tuple indexing.
# TODO: your code here
lst = []
lst = input('shopping items: ').split(',')
print(lst)
tpl = tuple(lst)
print(tpl[2])