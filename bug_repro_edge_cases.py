import sys
import os

# Add the parent directory to the system path to import VarTracer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from VarTracer.VarTracer_Core import VarTracer
from VarTracer.ASTParser import DependencyTree
import json

def test_dependency_analysis():
    print("Starting VarTracer Dependency Analysis Test...")
    vt = VarTracer(clean_stdlib=True, verbose=True)
    vt.start()

    # --- Test Cases ---
    
    # 1. Standard Assignment
    a = 10
    b = a + 5

    # 2. Multi-line Statement (Should fail or be missed by ASTParser)
    c = (
        a + 
        b
    )

    # 3. List Comprehension with Scope Masking
    x = 100
    y = [x for x in range(3)] # 'x' inside is local to comprehension, outer 'x' should not be dependency? Or should be?
    # Actually in Python 3, list comp has its own scope. Analyzer should see inner x as local.

    # 4. Lambda Function
    f = lambda z: z + x # 'x' is from outer scope (100)

    # 5. Walrus Operator (Python 3.8+)
    if (n := len(y)) > 0:
        d = n

    # 6. Augmented Assignment
    a += 1

    # 7. Subscript Assignment
    l = [1, 2, 3]
    l[0] = 10

    # 8. Unpacking Assignment
    u, v = 1, 2

    # 9. Function Definition (Single Line)
    def add(p, q): return p + q

    # 10. Default Arguments
    def default_arg(k=a): pass

    # 11. Class Definition
    class MyClass:
        cm = a
        def __init__(self):
            self.prop = b

    # 12. Import (Should be ignored or handled?)
    import math
    m = math.pi

    # 13. Exception Handling
    try:
        raise ValueError("error")
    except ValueError as e:
        err_msg = str(e)

    # 14. Nested Function
    def outer():
        o = 1
        def inner():
            return o
        return inner()
    
    res = outer()

    # --- End Test Cases ---

    vt.stop()

    print("\nGeneration Execution Stack...")
    exec_stack_json = vt.exec_stack_json(show_progress=False)

    print("\nParsing Dependency Tree...")
    dep_tree = DependencyTree(call_stack=json.dumps(exec_stack_json))
    dep_dic = dep_tree.parse_dependency()

    # Output relevant parts for inspection
    this_file = os.path.abspath(__file__)
    if this_file in dep_dic:
        print(f"\nDependencies for {this_file}:")
        print(json.dumps(dep_dic[this_file], indent=2, default=str))
    else:
        print(f"\nNo dependencies found for {this_file}")

if __name__ == "__main__":
    test_dependency_analysis()
