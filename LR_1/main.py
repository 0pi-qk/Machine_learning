def func1(num):
    if num < 10:
        return 0

    value = 1
    for i in str(num):
        value *= int(i)
    if value > 9:
        value = func1(value)

    return value


if __name__ == '__main__':
    value = func1(input('Enter number - '))
    print(value)
