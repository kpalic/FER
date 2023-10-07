def printaj(mat, ime):
    red,stup,value=mat
    print(f"{ime}:")
    for i in range(int(red)):
        for j in range(int(stup)):
            x=value.get((i,j),0)
            print("f{x:.2f}", end=' ')
        print()

import sys
if len(sys.argv)!=3:
      print("Nedostaje argument")
      sys.exit()
fileUlaz=sys.argv[1]
fileIzlaz=sys.argv[2]

mat=[]
with open(fileUlaz, 'r') as file:
   linija=file.readlines()
   i=0
   while i<len(linija):
      if(linija[i]=='\n'):
          i+=1
      ulaz=[]
      for x in linija[i].split():
          ulaz.append(int(x)) # ulaz = [int(x) for x in linija[i].split()]
      red=ulaz[0]
      stup=ulaz[1]
      matrica={}
      i+=1
      while i<len(linija) and linija[i]!='\n':
         print(f"usli smo s {linija}, {linija[i]}")
         red,stup,value=map(float, linija[i].split())
         matrica[int(red), int(stup)]=value
         i+=1
      mat.append((red,stup,matrica))
mat1=mat[0]
mat2=mat[1]
printaj(mat1,"A")
print(mat1)
print()
printaj(mat2,"B")
print(mat2)
print()

red1, stup1, value1 =mat1
red2, stup2, value2 =mat2

if stup1!=red2:
   print("Nekompaktibilne matrice.")
   sys.exit()

rezultat={}
for i in range(int(red1)):
    for j in range(int(stup2)):
        x=0
        for k in range(int(stup1)):
            x+=value1.get((i,k),0) * value2.get((k,j),0)
        if x!=0:
            rezultat[(i,j)]=x

rezultatMnozenja= red1, stup2, rezultat
printaj(rezultatMnozenja, "A*B")
print()

redI, stupI, valueI = rezultatMnozenja
with open(fileIzlaz, 'w') as file:
    file.write(f"{redI} {stupI}\n")
    for (redI, stupI), valueI in valueI.items():
        file.write(f"{redI} {stupI} {valueI:.2f}\n")
    file.write("\n")
print(f"Rezultat zapisan u datoteku {fileIzlaz}")