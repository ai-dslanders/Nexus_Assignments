def greet(name: str, times: int = 1) -> None:
    """Print `name`, capitalised, exactly `times` times on one line."""
    # TODO: your code here
    nam = []
    for i in range(times):
        nam.append(name.capitalize())
    print(*nam)


greet("alice")          # Alice
greet("bob", times=3)   # Bob Bob Bob
