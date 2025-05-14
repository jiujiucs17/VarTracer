def recurse(level, var = 1):
    result = 0
    if level > 0:
        result += recurse(level - 1, var + 1)
        return result
    else:
        # print("level:", level)
        return result + var