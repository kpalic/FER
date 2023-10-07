# ZI SKRIPTNI JEZICI 2019/2020 – corona edition


# 1

# A = "abcDEFghi"
# a)
# A[3:6] = "JKL"
# TypeError: 'str' object does not support item assignment

# b)
# A.replace("DEF", "JKL")
# C:\Users\eaprlik\Desktop\SKRJEZ\python>py zi.py
# abcDEFghi
# PROMJENI SE ALI SE NIKOM NE PRIDODA, OSTANE U ZRAKU

# c)
# A.split("DEF").join("JKL")
# AttributeError: 'list' object has no attribute 'join'

# d)
# A = A[:3] + "JKL" + A[6:]
# # CORRECT
# print(A)

# 2.)Lista R sadrzi realnu komponentu kompleksnog broja a lista I imaginuarnu. 
# Kako dobiti kompleksni broj?

# a) C = [r + i *1j for (r,i) in ZIP(R,I)]


# # 4.)
# M = [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]]
# # Ako je a = [[M[j][i] for j in range(len(M))] for i in range(len(M[0])] 
# # sto se nalazi u a

# # print(len(M)) # broj redaka
# a = [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
# print(a)

# 6)

# A = ('A', 'C', 'B')
# A = tuple(sorted(A, reverse=False))
# # print(sorted(A))
# print(A)


# 7)
# a = ['A', 'B', 'C', 'D']
# a[1:2] = []
# print(a)


# import gej
# gej.a = 4
# gej.b = 5 
# print(gej.a, gej.b, gej.c, end = " =")

# import gej
# print(gej.a, gej.b, gej.c, end = " =")


# def func(a, b):
#     a['c'] = 3
#     b += 13


# x = {'a' : 1, 'b' : 2}
# y = 12
# print(func(x,y))
# print (x, y)

# def f(a, b):
#     a[1] = 'b'
#     b = [2]
#     print("a: ", a)
#     print("b: ", b)
#     return a + b


# # a) C = f("724", "657")
# #  File "C:\Users\eaprlik\Desktop\SKRJEZ\python\zi.py", line 75, in f
# #     a[1] = 'b'
# #     ~^^^
# # TypeError: 'str' object does not support item assignment

# C = f([7,5],(6,5))
# print (C)

a="OkoSokolovo"
b=a.split('O')
a=[0][0],join(b)
print(a)
print(b)

