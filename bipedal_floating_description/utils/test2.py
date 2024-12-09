class test2:
    def __init__(self):
        self.x = 5

    def m1(self):
        self.x+=1
        print(self.x)
        
class test1:
    def __init__(self):
        t = test2()
        t.m1()

T=test1()
