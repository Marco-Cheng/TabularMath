# test_verifier.py
from augmentation.verifier import compile_verifier, compile_generator, RNGShim

VER = """
def verifier(assign):
    a = int(assign['a']); b = int(assign['b'])
    if a <= b: return False, None
    return True, a - b
"""

GEN = """
def generator(rng):
    a = rng.randint(8, 60)
    b = rng.randint(1, a-1)
    return {'a': a, 'b': b}
"""

def main():
    ver = compile_verifier(VER)
    gen = compile_generator(GEN)
    assign = gen(RNGShim(123))
    ok, y = ver(assign)
    print("assign=", assign, "ok=", ok, "y=", y)

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # macOS/3.13 建议显式设置
    except RuntimeError:
        pass
    main()
