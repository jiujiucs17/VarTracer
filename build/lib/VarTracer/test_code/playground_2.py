from test_code import recurse

def add(x, y):
    """Add two numbers together."""
    return x + y

def simple_decorator(func):
    """简单装饰器"""
    def wrapper(*args, **kwargs):
        print("Function is being called")
        return func(*args, **kwargs)
    return wrapper

def test_dependency_analyzer():
    # 测试递归函数
    x = recurse.recurse(2)

    # 测试多行赋值
    y = x + 5

    # 测试多目标赋值
    a, b = 1 + x, 2

    # 测试函数调用
    a = add(a, b)

    a += x+y

    # 测试解构赋值
    c, *d, e = [x, y, a, b]

    # 测试列表解析
    squares = [i**2 for i in range(5)]

    # 测试字典解析
    square_dict = {i: i**2 for i in range(5)}

    # 测试闭包
    def outer():
        z = 20
        def inner():
            return z + 1
        return inner()

    # 测试上下文管理器
    with open('test.txt', 'w') as f:
        f.write('DependencyAnalyzer test')

    # 测试异常处理
    try:
        result = 1 / 0
    except ZeroDivisionError as e:
        error_message = str(e)

    # 测试动态代码
    exec("dynamic_var = 42")
    eval_result = eval("dynamic_var + 1")

    # 测试类和实例方法
    class MyClass:
        def __init__(self, value):
            self.value = value

        def multiply(self, factor):
            return self.value * factor

    obj = MyClass(a)
    multiplied_value = obj.multiply(5)

    # 测试装饰器
    @simple_decorator
    def decorated_function():
        return "decorated"

def main():
    # print('This is the main program.')
    # recurse.recurse(2)
    # var_1 = 1
    # var_2 = 2
    # var_3 = add(var_1, var_2)
    # print('The sum of {} and {} is {}'.format(var_1, var_2, var_3))

    # 调用测试函数
    test_dependency_analyzer()

if __name__ == '__main__':
    main()