import fire

def hello(name_1='', name_2=''):
  return "Hello %s and %s!" % (name_1, name_2)

if __name__ == '__main__':
    fire.Fire(hello)
