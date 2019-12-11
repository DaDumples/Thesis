
def main():
    test(thing = '1')


def test(**a):
    test2(**a)

def test2(**kwargs):
    print(kwargs)

if __name__ == '__main__':
    main()